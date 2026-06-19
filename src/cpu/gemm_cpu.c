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
void multiplicar_naive(int N, const float *A, const float *B, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float soma = 0.0f;

            for (int k = 0; k < N; k++) {
                soma += A[i*N + k] * B[k*N + j];
            }

            C[i*N + j] = soma;
        }
    }
}

void transpor_matriz(int N, const float *B, float *BT) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            BT[j*N + i] = B[i*N + j];
        }
    }
}

void multiplicar_transposed(int N, const float *A, const float *BT, float *C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float soma = 0.0f;

            for (int k = 0; k < N; k++) {
                soma += A[i*N + k] * BT[j*N + k];
            }

            C[i*N + j] = soma;
        }
    }
}

// TODO: Implementar multiplicar_blocked(int N, int BS, const float *A, const float *B, float *C)
// TODO: Implementar multiplicar_openmp(int N, const float *A, const float *B, float *C)
// TODO: Implementar multiplicar_blocked_openmp(int N, int BS, const float *A, const float *B, float *C)

int main(int argc, char **argv) {
    // TODO: Implementar a lógica de medição de tempo e execução dos experimentos.
    // Utilizar os argumentos: <versao> <N> <repeticoes> [BS]
    if (argc < 4) {
        printf("Uso: %s <versao> <N> <repeticoes> [BS]\n", argv[0]);
        printf("Versoes: naive, transposed\n");
        return 1;
    }

    const char *versao = argv[1];
    int N = atoi(argv[2]);
    int repeticoes = atoi(argv[3]);

    size_t bytes = sizeof(float) * N * N;

    srand(0);

    float *A = malloc(bytes);
    float *B = malloc(bytes);
    float *C = malloc(bytes);
    float *C_ref = malloc(bytes);
    float *BT = malloc(bytes);

    if (!A || !B || !C || !C_ref || !BT) {
        printf("Erro ao alocar memoria.\n");
        return 1;
    }

    preencher_matriz(N, A);
    preencher_matriz(N, B);

    zerar_matriz(N, C_ref);
    multiplicar_naive(N, A, B, C_ref);

    double tempo_total = 0.0;

    for (int r = 0; r < repeticoes; r++) {
        zerar_matriz(N, C);

        double inicio = obter_segundos();

        if (strcmp(versao, "naive") == 0) {
            multiplicar_naive(N, A, B, C);
        } 
        else if (strcmp(versao, "transposed") == 0) {
            transpor_matriz(N, B, BT);
            multiplicar_transposed(N, A, BT, C);
        } 
        else {
            printf("Versao nao implementada nesta issue: %s\n", versao);
            return 1;
        }

        double fim = obter_segundos();
        tempo_total += fim - inicio;
    }

    double tempo_medio = tempo_total / repeticoes;
    double erro = erro_maximo_absoluto(N, C, C_ref);
    double gflops = (2.0 * N * N * N) / (tempo_medio * 1e9);

    printf("version,N,repeats,avg_time_s,GFLOPS,max_error\n");
    printf("%s,%d,%d,%.6f,%.6f,%.8f\n",
           versao, N, repeticoes, tempo_medio, gflops, erro);

    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(BT);

    return 0;
}
