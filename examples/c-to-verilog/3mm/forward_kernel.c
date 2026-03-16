#pragma HLS_interface P0 m_axi direct
#pragma HLS_interface P1 m_axi direct
#pragma HLS_interface P2 m_axi direct
#pragma HLS_interface P3 m_axi direct
#pragma HLS_interface P4 m_axi direct
#pragma HLS_interface P5 m_axi direct
#pragma HLS_interface P6 m_axi direct

#define S 10 // Number of elements in each dimension

#define NI S // Number of elements
#define NJ S // Number of elements
#define NK S // Number of elements
#define NL S // Number of elements
#define NM S // Number of elements

// Rename inputs/outputs
#define E P0
#define A P1
#define B P2
#define F P3
#define C P4
#define D P5
#define G P6

typedef float my_type;

/// Implements 3MM
/// E_size = NI,NJ
/// A_size = NI,NK
/// B_size = NK,NJ
/// F_size = NJ,NL
/// C_size = NJ,NM
/// D_size = NM,NL
/// G_size = NI,NL
void forward_kernel(my_type *restrict E, my_type *restrict A, my_type *restrict B,
                 my_type *restrict F, my_type *restrict C, my_type *restrict D,
                 my_type *restrict G) {
  int i, j, k;
  /* E := A*B */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++) {
      E[i * NJ + j] = 0.0;
      for (k = 0; k < NK; ++k)
        E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
    }
  /* F := C*D */
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++) {
      F[i * NL + j] = 0.0;
      for (k = 0; k < NM; ++k)
        F[i * NL + j] += C[i * NM + k] * D[k * NL + j];
    }
  /* G := E*F */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++) {
      G[i * NL + j] = 0.0;
      for (k = 0; k < NJ; ++k)
        G[i * NL + j] += E[i * NK + k] * F[k * NL + j];
    }
  return;
}