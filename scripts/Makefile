# Makefile para compilação do Trabalho Prático 3

CC = gcc
NVCC = nvcc

CFLAGS = -O3 -march=native -fopenmp -fopt-info-vec-optimized
CUDAFLAGS = -O3

CPU_SRC = ../src/cpu/gemm_cpu.c
GPU_SRC = ../src/gpu/gemm_cuda.cu

CPU_BIN = ../bin/gemm_cpu
GPU_BIN = ../bin/gemm_cuda

all: cpu gpu

cpu:
	mkdir -p ../bin
	$(CC) $(CFLAGS) $(CPU_SRC) -o $(CPU_BIN) 2> ../experiments/reports/vec_report.txt

gpu:
	mkdir -p ../bin
	$(NVCC) $(CUDAFLAGS) $(GPU_SRC) -o $(GPU_BIN)

clean:
	rm -rf ../bin
