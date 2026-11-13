// Testbench of mm.c: random-ish A, B, C, checks C against a plain loop.
#include <stdio.h>
#include <math.h>
#ifndef REP
#define REP 1
#endif
void mm(float *A, float *B, float *C);
int main()
{
   static float A[20][20], B[20][70], C[20][70], R[20][70];
   for(int i = 0; i < 20; i++)
      for(int k = 0; k < 20; k++)
         A[i][k] = (float)((i * 3 + k) % 7) - 3.0f;
   for(int k = 0; k < 20; k++)
      for(int j = 0; j < 70; j++)
         B[k][j] = (float)((k + 5 * j) % 11) * 0.5f;
   for(int i = 0; i < 20; i++)
      for(int j = 0; j < 70; j++)
         C[i][j] = R[i][j] = (float)(i - j) * 0.25f;
   for(int i = 0; i < 20; i++)
      for(int j = 0; j < 70; j++)
         for(int r = 0; r < REP; r++)
            for(int k = 0; k < 20; k++)
               R[i][j] += A[i][k] * B[k][j];
   mm(&A[0][0], &B[0][0], &C[0][0]);
   int bad = 0;
   for(int i = 0; i < 20; i++)
      for(int j = 0; j < 70; j++)
         bad += fabsf(C[i][j] - R[i][j]) > 1e-3f;
   printf("mismatches %d\n", bad);
   return bad != 0;
}
