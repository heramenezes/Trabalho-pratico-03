"""
Script para automatizar a execução dos experimentos em GPU.

Compila gemm_cuda.cu com nvcc e executa as versões naive e tiled para
múltiplos tamanhos de matriz, salvando os resultados em
experiments/results/resultados_gpu.csv.

Requer: NVIDIA CUDA Toolkit instalado e uma GPU disponível.
"""

import subprocess
import pandas as pd
import os
import sys
from io import StringIO

# ─── Parâmetros dos experimentos ────────────────────────────────────────────
Ns         = [128, 256, 512, 1024, 2048]
REPETICOES = 5
VERSOES    = ["naive", "tiled"]

# ─── Caminhos ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_GPU     = os.path.join(BASE_DIR, "src", "gpu", "gemm_cuda.cu")
BIN_DIR     = os.path.join(BASE_DIR, "bin")
BIN_GPU     = os.path.join(BIN_DIR, "gemm_cuda")
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")
OUT_CSV     = os.path.join(RESULTS_DIR, "resultados_gpu.csv")

os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def verificar_gpu():
    """Verifica se existe GPU disponível via nvidia-smi."""
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode != 0:
        print("[!] GPU não detectada. Verifique o driver NVIDIA / nvidia-smi.")
        sys.exit(1)
    print("[+] GPU detectada:")
    # Imprime apenas a linha do modelo da GPU
    for line in result.stdout.splitlines():
        if "%" in line or "MiB" in line:
            print("   ", line.strip())
            break


def compilar():
    """Compila gemm_cuda.cu com nvcc -O3."""
    print("[*] Compilando gemm_cuda.cu ...")
    cmd = ["nvcc", SRC_GPU, "-O3", "-o", BIN_GPU]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Erro de compilação CUDA:\n{result.stderr}")
        sys.exit(1)
    print(f"[+] Binário gerado em {BIN_GPU}")


def executar(versao, N, repeticoes):
    """Executa uma configuração CUDA e devolve um dict com os resultados."""
    cmd = [BIN_GPU, versao, str(N), str(repeticoes)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[!] Falha: {' '.join(cmd)}\n{result.stderr}")
        return None

    lines = result.stdout.strip().splitlines()
    if len(lines) < 2:
        return None

    df = pd.read_csv(StringIO("\n".join(lines[-2:])))
    return df.iloc[0].to_dict()


def main():
    verificar_gpu()
    compilar()

    experimentos = [(v, N) for N in Ns for v in VERSOES]
    rows = []
    total = len(experimentos)

    for idx, (versao, N) in enumerate(experimentos, 1):
        label = f"cuda_{versao} N={N}"
        print(f"[{idx}/{total}] {label} ...", end=" ", flush=True)

        row = executar(versao, N, REPETICOES)
        if row:
            rows.append(row)
            print(f"kernel={row.get('kernel_GFLOPS', '?'):.3f} GFLOP/s  "
                  f"total={row.get('total_GFLOPS', '?'):.3f} GFLOP/s")
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
