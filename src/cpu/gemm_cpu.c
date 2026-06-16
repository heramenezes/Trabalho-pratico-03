/**
 * @file gemm_cpu.c
 * @brief Implementação de multiplicação de matrizes (GEMM) em CPU.
 * 
 * Este arquivo deve conter as seguintes versões:
 * 1. Naive: Implementação básica com três loops aninhados.
 * 2. Transposed: Otimização de localidade de memória transpondo a matriz B.
 * 3. Blocked: Otimização de cache dividindo as matrizes em blocos.
 * 4. OpenMP: Paralelização das versões anteriores usando diretivas OpenMP.
 * 
 * TODO: Implementar as funções conforme as assinaturas abaixo.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifndef ALINHAMENTO
#define ALINHAMENTO 64
#endif

// Funções de utilidade
static double obter_segundos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float aleatorio_float() {
    return (float)rand() / (float)RAND_MAX;
}

static void preencher_matriz(int N, float *M) {
    for (int i = 0; i < N*N; i++) {
        M[i] = aleatorio_float();
    }
}

static void zerar_matriz(int N, float *M) {
    memset(M, 0, sizeof(float) * N * N);
}

static double erro_maximo_absoluto(int N, const float *A, const float *B) {
    double erro = 0.0;
    for (int i = 0; i < N*N; i++) {
        double e = fabs((double)A[i] - (double)B[i]);
        if (e > erro) erro = e;
    }
    return erro;
}

// Implementações das versões de multiplicação
// TODO: Implementar multiplicar_naive(int N, const float *A, const float *B, float *C)
// TODO: Implementar transpor_matriz(int N, const float *B, float *BT)
// TODO: Implementar multiplicar_transposed(int N, const float *A, const float *BT, float *C)
// TODO: Implementar multiplicar_blocked(int N, int BS, const float *A, const float *B, float *C)
// TODO: Implementar multiplicar_openmp(int N, const float *A, const float *B, float *C)
// TODO: Implementar multiplicar_blocked_openmp(int N, int BS, const float *A, const float *B, float *C)

int main(int argc, char **argv) {
    // TODO: Implementar a lógica de medição de tempo e execução dos experimentos.
    // Utilizar os argumentos: <versao> <N> <repeticoes> [BS]
    return 0;
}
