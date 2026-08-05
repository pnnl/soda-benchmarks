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

/// Debug print helper used by GenerateRankFunctionPass.
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void print_rank(int32_t rank);

// Print trace helper used by GenAddrFunctionPass.
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void print_trace(const char* trace);
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

/// Start a new counter group (for per-loop-group tracking).
///
/// \param groupId    Group identifier (assigned by the pass).
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void
sodaInstrDynamicCounterStartGroup(int64_t groupId);

/// Register function-name characters for a group header.
///
/// \param groupId    Group identifier (assigned by the pass).
/// \param charCode   ASCII character code to append.
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void
sodaInstrDynamicCounterSetGroupFunctionName(int64_t groupId, int64_t charCode);

/// Group counter flush entry point emitted by soda-instr-dynamic-counter.
///
/// \param groupId    Group identifier (assigned by the pass).
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void
sodaInstrDynamicCounterFlush(int64_t groupId);

/// Matrix access trace entry point emitted by soda-instr-matrix-access-trace.
///
/// \param baseAddr   Aligned memref base pointer represented as an i64.
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void
sodaInstrMarkMatrixAccessStarts(int64_t baseAddr);

/// Emit one summary line for a traced linalg op.
///
/// Unranked memref arguments follow the MLIR ABI: `(int64_t rank, void *ptr)`.
/// String arguments are rank-1 byte buffers.
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void
sodaInstrTraceLinalg(int64_t opId, int64_t opNameRank, void *opNamePtr,
                     int64_t numInputs, int64_t numOutputs, int64_t mapsRank,
                     void *mapsPtr, int64_t itersRank, void *itersPtr);

/// Emit one operand memref spec for a traced linalg op.
///
/// Unranked memref arguments follow the MLIR ABI: `(int64_t rank, void *ptr)`.
/// `dims` and `strides` are rank-1 i64 buffers.
extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void
sodaInstrTraceLinalgMemref(int64_t opId, int64_t operandKind,
                           int64_t operandIndex, int64_t baseAddr,
                           int64_t offset, int64_t dimsRank, void *dimsPtr,
                           int64_t stridesRank, void *stridesPtr);

#endif // SODAP_EXECUTIONENGINE_DYNAMICOPCOUNTERRUNTIME_H
