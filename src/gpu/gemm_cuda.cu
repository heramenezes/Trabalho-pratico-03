/**
 * @file gemm_cuda.cu
 * @brief Implementação de multiplicação de matrizes (GEMM) em GPU usando CUDA.
 *
 * Este arquivo contém as seguintes versões:
 * 1. CUDA Naive: Uma thread por elemento de C, acessando memória global diretamente.
 * 2. CUDA Tiled: Uso de memória compartilhada para otimizar acessos (blocos TILE×TILE).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE 16

// -----------------------------------------------------------------------
// Kernels CUDA
// -----------------------------------------------------------------------

__global__ void kernel_multiplicar_naive(int N, const float *A, const float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float soma = 0.0f;
        for (int k = 0; k < N; k++) {
            soma += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = soma;
    }
}

__global__ void kernel_multiplicar_tiled(int N, const float *A, const float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float soma = 0.0f;

    int num_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            soma += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = soma;
    }
}

// -----------------------------------------------------------------------
// Utilitários
// -----------------------------------------------------------------------

static void verificar_cuda(cudaError_t erro, const char *mensagem) {
    if (erro != cudaSuccess) {
        fprintf(stderr, "Erro CUDA em %s: %s\n", mensagem, cudaGetErrorString(erro));
        exit(1);
    }
}

static double obter_segundos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float aleatorio_float() {
    return (float)rand() / (float)RAND_MAX;
}

static void preencher_matriz(int N, float *M) {
    for (int i = 0; i < N * N; i++) {
        M[i] = aleatorio_float();
    }
}

static void zerar_matriz(int N, float *M) {
    memset(M, 0, sizeof(float) * N * N);
}

static double erro_maximo_absoluto(int N, const float *A, const float *B) {
    double erro = 0.0;
    for (int i = 0; i < N * N; i++) {
        double e = fabs((double)A[i] - (double)B[i]);
        if (e > erro) erro = e;
    }
    return erro;
}

extern "C" void multiplicar_naive(int N, const float *A, const float *B, float *C);

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <versao> <N> <repeticoes>\n", argv[0]);
        fprintf(stderr, "  versao: naive | tiled\n");
        return 1;
    }

    const char *versao  = argv[1];
    int N               = atoi(argv[2]);
    int repeticoes      = atoi(argv[3]);

    if (N <= 0 || repeticoes <= 0) {
        fprintf(stderr, "N e repeticoes devem ser positivos.\n");
        return 1;
    }

    if (strcmp(versao, "naive") != 0 && strcmp(versao, "tiled") != 0) {
        fprintf(stderr, "Versao desconhecida: %s  (use 'naive' ou 'tiled')\n", versao);
        return 1;
    }

    size_t bytes = sizeof(float) * N * N;

    // Alocação host
    float *h_A   = (float *)malloc(bytes);
    float *h_B   = (float *)malloc(bytes);
    float *h_C   = (float *)malloc(bytes);
    float *h_ref = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_ref) {
        fprintf(stderr, "Falha ao alocar memória no host.\n");
        return 1;
    }

    srand(42);
    preencher_matriz(N, h_A);
    preencher_matriz(N, h_B);

    // Alocação device
    float *d_A, *d_B, *d_C;
    verificar_cuda(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    verificar_cuda(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    verificar_cuda(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");

    dim3 bloco(TILE, TILE);
    dim3 grade((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    double t_h2d = 0.0, t_kernel = 0.0, t_d2h = 0.0;

    for (int r = 0; r < repeticoes; r++) {
        zerar_matriz(N, h_C);
        verificar_cuda(cudaMemset(d_C, 0, bytes), "cudaMemset d_C");

        // Transferência Host → Device
        double t0 = obter_segundos();
        verificar_cuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "H2D d_A");
        verificar_cuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "H2D d_B");
        verificar_cuda(cudaDeviceSynchronize(), "sync H2D");
        t_h2d += obter_segundos() - t0;

        // Execução do kernel
        t0 = obter_segundos();
        if (strcmp(versao, "naive") == 0) {
            kernel_multiplicar_naive<<<grade, bloco>>>(N, d_A, d_B, d_C);
        } else {
            kernel_multiplicar_tiled<<<grade, bloco>>>(N, d_A, d_B, d_C);
        }
        verificar_cuda(cudaGetLastError(), "launch kernel");
        verificar_cuda(cudaDeviceSynchronize(), "sync kernel");
        t_kernel += obter_segundos() - t0;

        // Transferência Device → Host
        t0 = obter_segundos();
        verificar_cuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "D2H d_C");
        verificar_cuda(cudaDeviceSynchronize(), "sync D2H");
        t_d2h += obter_segundos() - t0;
    }

    // Validação contra referência CPU (Issue #1)
    multiplicar_naive(N, h_A, h_B, h_ref);
    double erro = erro_maximo_absoluto(N, h_C, h_ref);

    printf("versao=%s N=%d repeticoes=%d t_h2d=%.6f t_kernel=%.6f t_d2h=%.6f erro_max=%.6e\n",
           versao, N, repeticoes,
           t_h2d    / repeticoes,
           t_kernel / repeticoes,
           t_d2h    / repeticoes,
           erro);

    // Liberação de memória
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
