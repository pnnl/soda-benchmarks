//===- DynamicOpCounterRuntime.h - Runtime support for op counters --------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares runtime entry points used by the dynamic operation-count
// instrumentation pass.
//
//===----------------------------------------------------------------------===//
#ifndef SODAP_EXECUTIONENGINE_DYNAMICOPCOUNTERRUNTIME_H
#define SODAP_EXECUTIONENGINE_DYNAMICOPCOUNTERRUNTIME_H

#ifdef _WIN32
#ifndef MLIR_SODAPINSTRRUNNERUTILS_EXPORT
#ifdef mlir_sodap_instr_runner_utils_EXPORTS
#define MLIR_SODAPINSTRRUNNERUTILS_EXPORT __declspec(dllexport)
#else
#define MLIR_SODAPINSTRRUNNERUTILS_EXPORT __declspec(dllimport)
#endif
#endif
#else
#define MLIR_SODAPINSTRRUNNERUTILS_EXPORT
#endif

#include <cstdint>

/// Entry point emitted by soda-instr-dynamic-op-counts-at-loop-bounds.
///
/// \\param run        Non-zero for loop-entry marker, zero for loop-exit marker.
/// \\param loopId     Stable id assigned to each loop by the pass.
/// \\param loads      Per-event load count delta.
/// \\param stores     Per-event store count delta.
/// \\param fpArith    Per-event floating-point arithmetic count delta.
/// \\param intArith   Per-event integer arithmetic count delta.
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void
sodaInstrCollectOpCounts(int64_t run, int64_t loopId, int64_t loads,
                         int64_t stores, int64_t fpArith, int64_t intArith);

/// Generic dynamic counter entry point emitted by soda-instr-dynamic-counter.
///
/// \param counterId  Counter kind identifier assigned by the pass.
/// \param delta      Increment amount (typically 1 per instrumented op).
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void
sodaInstrDynamicCounter(int64_t counterId, int64_t delta);

#endif // SODAP_EXECUTIONENGINE_DYNAMICOPCOUNTERRUNTIME_H
