#!/bin/bash

# Script de compilação manual para os programas CPU e GPU

mkdir -p ../bin

echo "Compilando versão CPU..."
gcc ../src/cpu/gemm_cpu.c -O3 -march=native -fopenmp -fopt-info-vec-optimized -o ../bin/gemm_cpu -lm 2> ../experiments/reports/vec_report.txt

if [ $? -eq 0 ]; then
    echo "CPU compilação OK."
else
    echo "Erro na compilação CPU."
    exit 1
fi

echo "Compilando objeto CPU para linkagem com GPU..."
gcc -c -DGEMM_CPU_LIB -O3 ../src/cpu/gemm_cpu.c -o ../bin/gemm_cpu_lib.o -lm

if [ $? -eq 0 ]; then
    echo "Objeto CPU compilado OK."
else
    echo "Erro ao compilar objeto CPU."
    exit 1
fi

echo "Compilando versão GPU (CUDA)..."
# Nota: Requer nvcc instalado e ambiente com GPU para compilar/executar
nvcc ../src/gpu/gemm_cuda.cu ../bin/gemm_cpu_lib.o -O3 -o ../bin/gemm_cuda -lm

if [ $? -eq 0 ]; then
    echo "GPU compilação OK."
else
    echo "Erro na compilação GPU."
    exit 1
fi
