// One compute step of gemm SMALL's matmul task, C[20][70] += A[20][20] * B[20][70],
// written in C so that Bambu's array partitioning (CSROA) accepts it. The
// buffers are local, like the accelerator's on-chip buffers: copy in, compute
// in MR x NR register blocks, copy out.
//   MR, NR      register block (default 10 x 7)
//   PA, PB, PC  cyclic banks: A by rows, B and C by columns (0 or unset: none)
//   REP         how many times the compute runs; compute cycles are
//               cycles(REP=2) - cycles(REP=1), which drops the copies
#define TI 20
#define TK 20
#define TJ 70
#ifndef MR
#define MR 10
#endif
#ifndef NR
#define NR 7
#endif
#ifndef REP
#define REP 1
#endif
#define STR(x) #x
#define PRAGMA(x) _Pragma(STR(x))

void mm(float A[TI * TK], float B[TK * TJ], float C[TI * TJ])
{
   float a[TI][TK], b[TK][TJ], c_[TI][TJ];
#if PA
   PRAGMA(HLS array_partition variable = a cyclic factor = PA dim = 1)
#endif
#if PB
   PRAGMA(HLS array_partition variable = b cyclic factor = PB dim = 2)
#endif
#if PC
   PRAGMA(HLS array_partition variable = c_ cyclic factor = PC dim = 2)
#endif
#pragma clang loop unroll(disable)
   for(int x = 0; x < TI * TK; x++)
      a[x / TK][x % TK] = A[x];
#pragma clang loop unroll(disable)
   for(int x = 0; x < TK * TJ; x++)
      b[x / TJ][x % TJ] = B[x];
#pragma clang loop unroll(disable)
   for(int x = 0; x < TI * TJ; x++)
      c_[x / TJ][x % TJ] = C[x];
#pragma clang loop unroll(disable)
   for(int r = 0; r < REP; r++)
#pragma clang loop unroll(disable)
   for(int ib = 0; ib < TI; ib += MR)
#pragma clang loop unroll(disable)
      for(int jb = 0; jb < TJ; jb += NR)
      {
         float c[MR][NR];
#pragma clang loop unroll(full)
         for(int ii = 0; ii < MR; ii++)
#pragma clang loop unroll(full)
            for(int jj = 0; jj < NR; jj++)
               c[ii][jj] = c_[ib + ii][jb + jj];
#pragma clang loop unroll(disable)
         for(int k = 0; k < TK; k++)
         {
#pragma clang loop unroll(full)
            for(int ii = 0; ii < MR; ii++)
#pragma clang loop unroll(full)
               for(int jj = 0; jj < NR; jj++)
                  c[ii][jj] += a[ib + ii][k] * b[k][jb + jj];
         }
#pragma clang loop unroll(full)
         for(int ii = 0; ii < MR; ii++)
#pragma clang loop unroll(full)
            for(int jj = 0; jj < NR; jj++)
               c_[ib + ii][jb + jj] = c[ii][jj];
      }
#pragma clang loop unroll(disable)
   for(int x = 0; x < TI * TJ; x++)
      C[x] = c_[x / TJ][x % TJ];
}
