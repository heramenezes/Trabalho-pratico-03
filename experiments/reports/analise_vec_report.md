# Análise do Relatório de Vetorização (vec_report.txt)

O relatório gerado pelo compilador GCC (`-fopt-info-vec-optimized`) indica quais loops e blocos de código puderam ser vetorizados utilizando instruções SIMD (Single Instruction, Multiple Data).

Analisando a saída gerada para o arquivo `gemm_cpu.c`, podemos observar o seguinte comportamento:

## 1. Versão Naive (`multiplicar_naive`)
```text
src\cpu\gemm_cpu.c:61:31: optimized: loop vectorized using 8 byte vectors
```
O compilador conseguiu aplicar apenas uma vetorização parcial/limitada (8 bytes = 2 floats simultâneos). Isso ocorre porque, no loop mais interno (`k`), o acesso à matriz `B` não é contíguo (`B[k*N + j]`). Os acessos pulam na memória a cada iteração, dificultando a vetorização eficiente.

## 2. Versão Transposed (`multiplicar_transposed`)
```text
src\cpu\gemm_cpu.c:83:31: optimized: loop vectorized using 64 byte vectors
src\cpu\gemm_cpu.c:83:31: optimized: loop vectorized using 32 byte vectors
```
Aqui o compilador conseguiu utilizar vetores muito maiores (64 bytes = 16 floats simultâneos, indicando o uso de instruções AVX-512 ou similar, a depender da arquitetura, e 32 bytes = 8 floats para loops residuais). Como a matriz `B` foi transposta e é acessada como `BT[j*N + k]`, ambos os acessos (`A` e `BT`) tornam-se sequenciais em relação a `k`. Isso maximiza o uso da cache e permite que o vetorizador atinja sua eficiência máxima.

## 3. Versões OpenMP e Blocked
```text
src\cpu\gemm_cpu.c:115:13: optimized: basic block part vectorized using 16 byte vectors
src\cpu\gemm_cpu.c:128:13: optimized: basic block part vectorized using 8 byte vectors
src\cpu\gemm_cpu.c:128:13: optimized: basic block part vectorized using 16 byte vectors
```
Para as versões OpenMP e Blocked (linhas ~115 e ~128), as operações internas sofrem as mesmas limitações da versão *naive* (acesso não-contíguo de `B`). O compilador realiza vetorização de partes do bloco básico ("basic block part vectorized") com vetores curtos (8 e 16 bytes), que não oferecem a mesma aceleração agressiva vista na versão *transposed*.

## Conclusão
A transposição da matriz de destino garante a localidade de memória no acesso contíguo. O compilador detecta isso e emite instruções que operam com múltiplos dados ao mesmo tempo, permitindo as vetorizações de 32 e 64 bytes apontadas pelo relatório. Para obter a máxima performance (mesmo usando blocagem ou OpenMP), o ideal é alinhar as otimizações de nível de macro (tiling, threads) com as otimizações de micro (localidade de acesso contíguo e vetorização).
