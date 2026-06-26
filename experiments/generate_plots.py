"""
Script para gerar automaticamente os gráficos do Trabalho Prático 3.

Produz na pasta experiments/results/plots/:
  - gflops_cpu_por_N.png      : GFLOP/s das versões CPU por tamanho de matriz
  - speedup_cpu.png            : Speedup relativo ao naive para versões CPU
  - eficiencia_openmp.png      : Eficiência paralela das versões OpenMP
  - tempo_gpu_kernel.png       : Tempo de kernel CUDA (naive vs tiled)
  - gflops_gpu.png             : GFLOP/s kernel CUDA por N
  - overhead_transferencia.png : Breakdown H2D / Kernel / D2H (GPU)
  - speedup_gpu_vs_cpu.png     : Comparação GPU tiled vs CPU melhor versão
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─── Caminhos ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "experiments", "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
CPU_CSV     = os.path.join(RESULTS_DIR, "resultados_cpu.csv")
GPU_CSV     = os.path.join(RESULTS_DIR, "resultados_gpu.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "axes.spines.top":  False,
    "axes.spines.right":False,
}
plt.rcParams.update(STYLE)
COLORS = plt.cm.tab10.colors


def salvar(fig, nome):
    path = os.path.join(PLOTS_DIR, nome)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [+] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos CPU
# ─────────────────────────────────────────────────────────────────────────────

def graficos_cpu(cpu):
    Ns = sorted(cpu["N"].unique())

    # 1. GFLOP/s por versão e N ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(Ns), figsize=(5 * len(Ns), 5), sharey=False)
    if len(Ns) == 1:
        axes = [axes]

    for ax, N in zip(axes, Ns):
        df = cpu[cpu["N"] == N].copy()
        df["label"] = (
            df["version"]
            + df.apply(
                lambda r: f"\nBS={int(r['BS'])}" if str(r["BS"]) not in ("N/A", "nan", "") and not pd.isna(r["BS"]) else "",
                axis=1,
            )
            + df.apply(
                lambda r: f"\nT={int(r['threads'])}" if str(r["threads"]) not in ("N/A", "nan", "1", "") and not pd.isna(r["threads"]) else "",
                axis=1,
            )
        )
        bars = ax.bar(range(len(df)), df["GFLOPS"], color=COLORS[:len(df)], edgecolor="grey", linewidth=0.5)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["label"], fontsize=7, rotation=45, ha="right")
        ax.set_title(f"N = {N}", fontweight="bold")
        ax.set_ylabel("GFLOP/s")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    fig.suptitle("Desempenho (GFLOP/s) por Versão CPU", fontsize=13, fontweight="bold", y=1.02)
    salvar(fig, "gflops_cpu_por_N.png")

    # 2. Speedup relativo ao naive ────────────────────────────────────────────
    versoes_fixas = ["naive", "transposed", "blocked", "openmp", "blocked_openmp"]
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(Ns))
    width = 0.15
    offsets = np.linspace(-(len(versoes_fixas)-1)/2, (len(versoes_fixas)-1)/2, len(versoes_fixas))

    for i, versao in enumerate(versoes_fixas):
        speedups = []
        for N in Ns:
            naive_row = cpu[(cpu["N"] == N) & (cpu["version"] == "naive")]
            if naive_row.empty:
                speedups.append(0); continue
            t_naive = naive_row["avg_time_s"].values[0]

            # Pega melhor tempo da versão
            subset = cpu[(cpu["N"] == N) & (cpu["version"] == versao)]
            if subset.empty:
                speedups.append(0); continue
            t_best = subset["avg_time_s"].min()
            speedups.append(t_naive / t_best if t_best > 0 else 0)

        ax.bar(x + offsets[i] * width, speedups, width=width,
               label=versao, color=COLORS[i], edgecolor="grey", linewidth=0.5)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="baseline (naive)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={N}" for N in Ns])
    ax.set_ylabel("Speedup (vs naive)")
    ax.set_title("Speedup das Versões CPU em relação à Versão Naive", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    salvar(fig, "speedup_cpu.png")

    # 3. Eficiência OpenMP ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for N in Ns:
        subset_omp = cpu[(cpu["N"] == N) & (cpu["version"] == "openmp")].copy()
        if subset_omp.empty:
            continue
        subset_omp = subset_omp.sort_values("threads")
        t1_row = subset_omp[subset_omp["threads"] == 1]
        if t1_row.empty:
            continue
        t1 = t1_row["avg_time_s"].values[0]
        threads = subset_omp["threads"].values
        speedups = t1 / subset_omp["avg_time_s"].values
        eficiencia = speedups / threads * 100
        ax.plot(threads, eficiencia, marker="o", label=f"N={N}")

    ax.axhline(100, color="black", linestyle="--", linewidth=0.8, label="Eficiência ideal (100%)")
    ax.set_xlabel("Número de Threads")
    ax.set_ylabel("Eficiência (%)")
    ax.set_title("Eficiência Paralela — OpenMP", fontweight="bold")
    ax.legend()
    salvar(fig, "eficiencia_openmp.png")


# ─────────────────────────────────────────────────────────────────────────────
# Gráficos GPU
# ─────────────────────────────────────────────────────────────────────────────

def graficos_gpu(gpu):
    Ns = sorted(gpu["N"].unique())

    # 4. Tempo de kernel CUDA ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, versao in enumerate(gpu["version"].unique()):
        df = gpu[gpu["version"] == versao].sort_values("N")
        ax.plot(df["N"], df["kernel_ms"], marker="o", label=versao, color=COLORS[i])
    ax.set_xlabel("Tamanho da Matriz (N)")
    ax.set_ylabel("Tempo do Kernel (ms)")
    ax.set_title("Tempo do Kernel CUDA por Tamanho de Matriz", fontweight="bold")
    ax.legend()
    salvar(fig, "tempo_gpu_kernel.png")

    # 5. GFLOP/s kernel GPU ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, versao in enumerate(gpu["version"].unique()):
        df = gpu[gpu["version"] == versao].sort_values("N")
        ax.plot(df["N"], df["kernel_GFLOPS"], marker="o", label=versao, color=COLORS[i])
    ax.set_xlabel("Tamanho da Matriz (N)")
    ax.set_ylabel("GFLOP/s (Kernel)")
    ax.set_title("Desempenho do Kernel CUDA (GFLOP/s)", fontweight="bold")
    ax.legend()
    salvar(fig, "gflops_gpu.png")

    # 6. Breakdown H2D / Kernel / D2H ────────────────────────────────────────
    for versao in gpu["version"].unique():
        df = gpu[gpu["version"] == versao].sort_values("N")
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(df))
        ax.bar(x, df["h2d_ms"],    label="H2D",    color=COLORS[0])
        ax.bar(x, df["kernel_ms"], bottom=df["h2d_ms"], label="Kernel", color=COLORS[1])
        ax.bar(x, df["d2h_ms"],
               bottom=df["h2d_ms"] + df["kernel_ms"],
               label="D2H", color=COLORS[2])
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={n}" for n in df["N"]])
        ax.set_ylabel("Tempo (ms)")
        ax.set_title(f"Breakdown de Tempo — {versao}", fontweight="bold")
        ax.legend()
        salvar(fig, f"overhead_transferencia_{versao}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Comparação CPU vs GPU
# ─────────────────────────────────────────────────────────────────────────────

def graficos_comparacao(cpu, gpu):
    Ns_comuns = sorted(set(cpu["N"].unique()) & set(gpu["N"].unique()))
    if not Ns_comuns:
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    # Melhor CPU (blocked_openmp com maior GFLOPS)
    cpu_best = []
    for N in Ns_comuns:
        subset = cpu[cpu["N"] == N]
        cpu_best.append(subset["GFLOPS"].max() if not subset.empty else 0)

    # GPU tiled — kernel GFLOPS
    tiled_vals = []
    for N in Ns_comuns:
        subset = gpu[(gpu["version"].str.contains("tiled")) & (gpu["N"] == N)]
        tiled_vals.append(subset["kernel_GFLOPS"].values[0] if not subset.empty else 0)

    # GPU naive — kernel GFLOPS
    naive_vals = []
    for N in Ns_comuns:
        subset = gpu[(gpu["version"].str.contains("naive")) & (gpu["N"] == N)]
        naive_vals.append(subset["kernel_GFLOPS"].values[0] if not subset.empty else 0)

    x = np.arange(len(Ns_comuns))
    width = 0.25
    ax.bar(x - width, cpu_best,    width=width, label="CPU (melhor)",   color=COLORS[0])
    ax.bar(x,         naive_vals,  width=width, label="GPU naive (kernel)", color=COLORS[1])
    ax.bar(x + width, tiled_vals,  width=width, label="GPU tiled (kernel)", color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={N}" for N in Ns_comuns])
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Comparação CPU (melhor) vs GPU — GFLOP/s", fontweight="bold")
    ax.legend()
    salvar(fig, "speedup_gpu_vs_cpu.png")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    cpu_ok = os.path.exists(CPU_CSV)
    gpu_ok = os.path.exists(GPU_CSV)

    if not cpu_ok and not gpu_ok:
        print("[!] Nenhum CSV de resultados encontrado. Execute run_cpu_bench.py e/ou run_gpu_bench.py primeiro.")
        sys.exit(1)

    print(f"[*] Gerando gráficos em {PLOTS_DIR}/\n")

    if cpu_ok:
        cpu = pd.read_csv(CPU_CSV)
        print("[*] Gráficos CPU:")
        graficos_cpu(cpu)

    if gpu_ok:
        gpu = pd.read_csv(GPU_CSV)
        print("[*] Gráficos GPU:")
        graficos_gpu(gpu)

    if cpu_ok and gpu_ok:
        print("[*] Gráficos Comparativos:")
        graficos_comparacao(cpu, gpu)

    print("\n[+] Todos os gráficos gerados com sucesso!")


if __name__ == "__main__":
    main()
