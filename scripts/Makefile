# Makefile para compilação do Trabalho Prático 3

CC = gcc
NVCC = nvcc

CFLAGS = -O3 -march=native -fopenmp -fopt-info-vec-optimized
CUDAFLAGS = -O3

CPU_SRC = ../src/cpu/gemm_cpu.c
GPU_SRC = ../src/gpu/gemm_cuda.cu

CPU_BIN = ../bin/gemm_cpu
GPU_BIN = ../bin/gemm_cuda
CPU_OBJ = ../bin/gemm_cpu_lib.o

all: cpu gpu

cpu:
	mkdir -p ../bin
	$(CC) $(CFLAGS) $(CPU_SRC) -o $(CPU_BIN) -lm 2> ../experiments/reports/vec_report.txt

$(CPU_OBJ):
	mkdir -p ../bin
	$(CC) -c -DGEMM_CPU_LIB -O3 $(CPU_SRC) -o $(CPU_OBJ) -lm

gpu: $(CPU_OBJ)
	mkdir -p ../bin
	$(NVCC) $(CUDAFLAGS) $(GPU_SRC) $(CPU_OBJ) -o $(GPU_BIN) -lm

clean:
	rm -rf ../bin
