"""
Script para automatizar a execução dos experimentos em CPU.

Compila o gemm_cpu.c e executa todas as versões para múltiplos tamanhos
de matriz e configurações de threads/block-size, salvando os resultados
em experiments/results/resultados_cpu.csv.
"""

import subprocess
import pandas as pd
import os
import sys
from io import StringIO

# ─── Parâmetros dos experimentos ────────────────────────────────────────────
Ns        = [128, 256, 512, 1024]   # tamanhos de matriz
REPETICOES = 5                       # repetições por configuração
THREADS    = [1, 2, 4, 8]           # contagens de threads OpenMP
BLOCK_SIZES = [16, 32, 64]          # tamanhos de bloco para versão blocked

# ─── Caminhos ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CPU     = os.path.join(BASE_DIR, "src", "cpu", "gemm_cpu.c")
BIN_DIR     = os.path.join(BASE_DIR, "bin")
BIN_CPU     = os.path.join(BIN_DIR, "gemm_cpu")
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "resultados_cpu.csv")
VEC_REPORT  = os.path.join(BASE_DIR, "experiments", "reports", "vec_report.txt")

os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def compilar():
    """Compila gemm_cpu.c com -O3, -march=native e -fopenmp."""
    print("[*] Compilando gemm_cpu.c ...")
    cmd = [
        "gcc", SRC_CPU,
        "-O3", "-march=native", "-fopenmp",
        "-fopt-info-vec-optimized",
        "-lm",
        "-o", BIN_CPU,
    ]
    with open(VEC_REPORT, "w") as vec_out:
        result = subprocess.run(cmd, stderr=vec_out, text=True)

    if result.returncode != 0:
        print(f"[!] Erro de compilação. Verifique {SRC_CPU}.")
        sys.exit(1)
    print(f"[+] Binário gerado em {BIN_CPU}")
    print(f"[+] Relatório de vetorização em {VEC_REPORT}")


def executar(versao, N, repeticoes, bs=None, threads=None):
    """Executa uma configuração e devolve um dict com os resultados."""
    cmd = [BIN_CPU, versao, str(N), str(repeticoes)]
    if bs is not None:
        cmd.append(str(bs))

    env = os.environ.copy()
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        print(f"[!] Falha: {' '.join(cmd)}\n{result.stderr}")
        return None

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        return None

    df = pd.read_csv(StringIO("\n".join(lines[-2:])))
    return df.iloc[0].to_dict()


def main():
    compilar()

    experimentos = []

    for N in Ns:
        # Versão naive (sem threads)
        experimentos.append(("naive",          N, None, None))
        # Versão transposed (sem threads)
        experimentos.append(("transposed",     N, None, None))
        # Versão blocked: vários block sizes, sem OpenMP
        for bs in BLOCK_SIZES:
            experimentos.append(("blocked",    N, bs,   None))
        # Versão openmp: vários contagens de threads, sem block size
        for th in THREADS:
            experimentos.append(("openmp",     N, None, th))
        # Versão blocked_openmp: block size fixo + vários threads
        for th in THREADS:
            experimentos.append(("blocked_openmp", N, 32, th))

    rows = []
    total = len(experimentos)
    for idx, (versao, N, bs, th) in enumerate(experimentos, 1):
        label = f"{versao} N={N}"
        if bs:   label += f" BS={bs}"
        if th:   label += f" T={th}"
        print(f"[{idx}/{total}] {label} ...", end=" ", flush=True)

        row = executar(versao, N, REPETICOES, bs=bs, threads=th)
        if row:
            # Garante colunas extras para consistência no CSV
            row.setdefault("BS",      bs  if bs  is not None else "N/A")
            row.setdefault("threads", th  if th  is not None else 1)
            rows.append(row)
            print(f"{row.get('GFLOPS', '?'):.3f} GFLOP/s")
        else:
            print("ERRO")

    if not rows:
        print("[!] Nenhum resultado coletado.")
        sys.exit(1)

    df_final = pd.DataFrame(rows)
    df_final.to_csv(OUT_CSV, index=False)
    print(f"\n[+] Resultados salvos em {OUT_CSV}")
    print(df_final.to_string(index=False))


if __name__ == "__main__":
    main()
