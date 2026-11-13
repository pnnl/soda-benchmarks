/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * ESPLink loads a raw binary, which contains no .sbss/.bss bytes. DDR may
 * still hold the preceding application's data. Initialize these sections
 * before ESP's _init initializes TLS and calls main. The ESP crt has already
 * parked secondary harts, established gp and set up the stack at this point.
 * Compile ESP's crt.S with -D_init=esp_init_bss and use the bounds defined by
 * run-system.sh. This avoids --wrap aliases during RISC-V linker relaxation.
 */
  .section .text.esp_bss_init,"ax",@progbits
  .balign 4
  .globl esp_init_bss
  .type esp_init_bss, @function
esp_init_bss:
  la t0, __esp_bss_start
  la t1, __esp_bss_end
  bgeu t0, t1, 2f
1:
  sd zero, 0(t0)
  addi t0, t0, 8
  bltu t0, t1, 1b
2:
  /* Preserve a0/a1 (hart ID and hart count) for the original _init. */
  tail _init
  .size esp_init_bss, .-esp_init_bss
