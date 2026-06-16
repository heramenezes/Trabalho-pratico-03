#!/bin/bash

# Script de compilação manual para os programas CPU e GPU

echo "Compilando versão CPU..."
gcc ../src/cpu/gemm_cpu.c -O3 -march=native -fopenmp -fopt-info-vec-optimized -o ../bin/gemm_cpu 2> ../experiments/reports/vec_report.txt

if [ $? -eq 0 ]; then
    echo "CPU compilação OK."
else
    echo "Erro na compilação CPU."
fi

echo "Compilando versão GPU (CUDA)..."
# Nota: Requer nvcc instalado e ambiente com GPU para compilar/executar
nvcc ../src/gpu/gemm_cuda.cu -O3 -o ../bin/gemm_cuda

if [ $? -eq 0 ]; then
    echo "GPU compilação OK."
else
    echo "Erro na compilação GPU."
fi
