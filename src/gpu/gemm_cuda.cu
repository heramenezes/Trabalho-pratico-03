/**
 * @file gemm_cuda.cu
 * @brief Implementação de multiplicação de matrizes (GEMM) em GPU usando CUDA.
 *
 * Versões implementadas:
 * 1. CUDA Naive: Uma thread por elemento de C, acessando memória global diretamente.
 * 2. CUDA Tiled: Uso de memória compartilhada para otimizar acessos.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define TILE 16

static float rand_float() {
    return (float)rand() / (float)RAND_MAX;
}

static void fill_matrix(int N, float *M) {
    for (int i = 0; i < N*N; i++) M[i] = rand_float();
}

static void gemm_cpu_ref(int N, const float *A, const float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

static double max_abs_error(int N, const float *A, const float *B) {
    double err = 0.0;
    for (int i = 0; i < N*N; i++) {
        double e = fabs((double)A[i] - (double)B[i]);
        if (e > err) err = e;
    }
    return err;
}

/**
 * Kernel Naive: cada thread calcula um único elemento C[row][col].
 * Acessa diretamente a memória global sem otimização de localidade.
 */
__global__ void gemm_cuda_naive(int N, const float *A, const float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

/**
 * Kernel Tiled: usa memória compartilhada (shared memory) para reduzir
 * acessos redundantes à memória global. Cada bloco de threads carrega
 * um tile de A e um tile de B na shared memory e reutiliza esses dados.
 */
__global__ void gemm_cuda_tiled(int N, const float *A, const float *B, float *C) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; t++) {
        // Carrega tile de A na shared memory
        int a_col = t * TILE + threadIdx.x;
        if (row < N && a_col < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Carrega tile de B na shared memory
        int b_row = t * TILE + threadIdx.y;
        if (b_row < N && col < N)
            sB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        // Sincroniza para garantir que o tile foi carregado por todas as threads
        __syncthreads();

        // Produto parcial usando os dados em shared memory
        for (int k = 0; k < TILE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Sincroniza antes de carregar o próximo tile
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("Usage: %s <naive|tiled> <N> <repeats>\n", argv[0]);
        return 1;
    }

    const char *version = argv[1];
    int N = atoi(argv[2]);
    int repeats = atoi(argv[3]);
    size_t bytes = sizeof(float) * N * N;

    srand(0);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    fill_matrix(N, h_A);
    fill_matrix(N, h_B);
    gemm_cpu_ref(N, h_A, h_B, h_ref);

    float *d_A, *d_B, *d_C;
    check_cuda(cudaMalloc(&d_A, bytes), "cudaMalloc A");
    check_cuda(cudaMalloc(&d_B, bytes), "cudaMalloc B");
    check_cuda(cudaMalloc(&d_C, bytes), "cudaMalloc C");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float h2d_ms_total = 0.0f;
    float kernel_ms_total = 0.0f;
    float d2h_ms_total = 0.0f;

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    for (int r = 0; r < repeats; r++) {
        float ms;

        // H2D
        cudaEventRecord(start);
        check_cuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "copy A H2D");
        check_cuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "copy B H2D");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        h2d_ms_total += ms;

        // Kernel
        cudaEventRecord(start);
        if (strcmp(version, "naive") == 0) {
            gemm_cuda_naive<<<grid, block>>>(N, d_A, d_B, d_C);
        } else {
            gemm_cuda_tiled<<<grid, block>>>(N, d_A, d_B, d_C);
        }
        check_cuda(cudaGetLastError(), "kernel launch");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        kernel_ms_total += ms;

        // D2H
        cudaEventRecord(start);
        check_cuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "copy C D2H");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        d2h_ms_total += ms;
    }

    float h2d_ms    = h2d_ms_total    / repeats;
    float kernel_ms = kernel_ms_total / repeats;
    float d2h_ms    = d2h_ms_total    / repeats;
    float total_ms  = h2d_ms + kernel_ms + d2h_ms;

    double kernel_gflops = (2.0 * N * N * N) / ((kernel_ms / 1000.0) * 1e9);
    double total_gflops  = (2.0 * N * N * N) / ((total_ms  / 1000.0) * 1e9);
    double err = max_abs_error(N, h_C, h_ref);

    printf("version,N,repeats,h2d_ms,kernel_ms,d2h_ms,total_ms,kernel_GFLOPS,total_GFLOPS,max_abs_error\n");
    printf("cuda_%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f\n",
           version, N, repeats, h2d_ms, kernel_ms, d2h_ms, total_ms,
           kernel_gflops, total_gflops, err);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
