# Trabalho Prático 3 — Otimização Paralela de GEMM

Este repositório contém a implementação e análise de desempenho da operação de multiplicação de matrizes (GEMM) otimizada para CPU e GPU, desenvolvida para a disciplina de **Arquitetura de Computadores**.

## Estrutura do Projeto

```text
/
├── bin/                    # Executáveis compilados
├── docs/                   # Relatório final e documentação
├── experiments/            # Scripts de benchmark e resultados
│   ├── reports/            # Relatórios de otimização (vetorização)
│   ├── results/            # Dados brutos dos experimentos (CSV)
│   ├── run_cpu_bench.py    # Automatização de testes em CPU
│   └── run_gpu_bench.py    # Automatização de testes em GPU
├── notebooks/              # Notebook Jupyter/Colab original
├── scripts/                # Scripts de compilação (Makefile, shell)
├── src/                    # Código fonte
│   ├── cpu/                # Implementações C (Naive, OpenMP, etc.)
│   └── gpu/                # Implementações CUDA (Naive, Tiled)
└── README.md               # Este arquivo
```

## Como Usar

### Pré-requisitos
- Compilador GCC com suporte a OpenMP.
- NVIDIA CUDA Toolkit (para a versão GPU).
- Python 3 com as bibliotecas `pandas` e `matplotlib` para análise de dados.

### Compilação
Para compilar as versões CPU e GPU:
```bash
cd scripts
make all
```

### Execução de Experimentos
Para rodar os benchmarks e coletar dados:
```bash
python experiments/run_cpu_bench.py
python experiments/run_gpu_bench.py
```
