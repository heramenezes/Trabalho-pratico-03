/**
 * @file gemm_cuda.cu
 * @brief Implementação de multiplicação de matrizes (GEMM) em GPU usando CUDA.
 * 
 * Este arquivo deve conter as seguintes versões:
 * 1. CUDA Naive: Uma thread por elemento de C, acessando memória global diretamente.
 * 2. CUDA Tiled: Uso de memória compartilhada para otimizar acessos.
 * 
 * TODO: Implementar os kernels CUDA e a lógica de transferência de dados.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE 16

// Kernels CUDA
// TODO: Implementar __global__ void kernel_multiplicar_naive(int N, const float *A, const float *B, float *C)
// TODO: Implementar __global__ void kernel_multiplicar_tiled(int N, const float *A, const float *B, float *C)

static void verificar_cuda(cudaError_t erro, const char *mensagem) {
    if (erro != cudaSuccess) {
        fprintf(stderr, "Erro CUDA em %s: %s\n", mensagem, cudaGetErrorString(erro));
        exit(1);
    }
}

int main(int argc, char **argv) {
    // TODO: Implementar:
    // 1. Alocação de memória (Host e Device).
    // 2. Inicialização das matrizes.
    // 3. Medição de tempos: H2D, Kernel, D2H.
    // 4. Validação do resultado final.
    return 0;
}
