//===- DynamicOpCounterRuntime.cpp - Runtime support for op counters ------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/ExecutionEngine/DynamicOpCounterRuntime.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <limits>
#include "llvm/ADT/ArrayRef.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include <string>

namespace {
struct CounterState {
  int64_t loads = 0;
  int64_t stores = 0;
  int64_t fpArith = 0;
  int64_t intArith = 0;

  void add(int64_t l, int64_t s, int64_t fp, int64_t in) {
    loads += l;
    stores += s;
    fpArith += fp;
    intArith += in;
  }

  void reset() {
    loads = 0;
    stores = 0;
    fpArith = 0;
    intArith = 0;
  }
};

std::unordered_map<int64_t, CounterState> gCounterByLoopId;
std::unordered_map<int64_t, int64_t> gDynamicCounters;
std::unordered_map<int64_t, int64_t> gGroupDynamicCounters;
std::unordered_map<int64_t, std::string> gGroupFunctionNames;
std::unordered_map<std::string, std::unordered_map<int64_t, int64_t>>
    gFunctionDynamicCounters;
std::vector<int64_t> gActiveLoopStack;
bool gAtExitRegistered = false;
int64_t gCurrentGroupId = -1;
constexpr int64_t kCacheLineSizeElements = 4;

extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void print_rank(int32_t rank) {
  std::printf("linalg.generic rank = %d\n", rank);
}

extern "C" MLIR_SODAPINSTRRUNNERUTILS_EXPORT void print_trace(const char* trace) {
  std::cout << "Gen addr: " << trace << std::endl;
}
const char *linalgOperandKindName(int64_t operandKind) {
  switch (operandKind) {
  case 0:
    return "in";
  case 1:
    return "out";
  default:
    return "unknown";
  }
}

void printLoopSummary(int64_t loopId, const CounterState &state) {
  std::cout << "SODA_COUNTER loop=" << loopId << " loads=" << state.loads
            << " stores=" << state.stores << " fp=" << state.fpArith
            << " int=" << state.intArith << std::endl;
}

void printAllLoopSummaries() {
  std::vector<int64_t> loopIds;
  loopIds.reserve(gCounterByLoopId.size());
  for (const auto &it : gCounterByLoopId)
    loopIds.push_back(it.first);

  std::sort(loopIds.begin(), loopIds.end());
  for (int64_t loopId : loopIds)
    printLoopSummary(loopId, gCounterByLoopId[loopId]);
}

const char *dynamicCounterName(int64_t counterId) {
    static constexpr std::array<const char *, 6> names = {{
        "memref.load",
        "memref.store",
        "arith.int",
        "arith.float",
        "scf",
        "affine"
    }};

    if (counterId >= 0 && counterId < static_cast<int64_t>(names.size())) {
        return names[counterId];
    }
    return "unknown";
}

int64_t clampPositive(int64_t value) {
  return value > 0 ? value : 1;
}

void printAddrLinear(uint64_t baseAddr, int64_t totalElements, int64_t step,
                     int64_t elementBytes) {
  for (int64_t i = 0; i < totalElements; i += step) {
    uint64_t addr =
        baseAddr + static_cast<uint64_t>(i) * static_cast<uint64_t>(elementBytes);
    std::cout << "0x" << std::hex << addr << std::dec << "\t element[" << i
              << "]" << std::endl;
  }
}

std::string formatI64Descriptor(uint64_t ptrAsInt, int64_t count) {
  std::ostringstream os;
  os << '[';
  if (ptrAsInt == 0 || count <= 0) {
    os << ']';
    return os.str();
  }

  const int64_t *data = reinterpret_cast<const int64_t *>(ptrAsInt);
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0)
      os << ", ";
    os << data[i];
  }
  os << ']';
  return os.str();
}


template <typename T>
std::string formatVector(const DynamicMemRefType<T> &memref) {
  std::ostringstream os;
  os << '[';
  int64_t count = 0;
  if (memref.rank > 0 && memref.sizes)
    count = memref.sizes[0];
  for (int64_t i = 0; i < count; ++i) {
    if (i != 0)
      os << ", ";
    os << static_cast<int64_t>(memref.data[memref.offset + i]);
  }
  os << ']';
  return os.str();
}

std::string stringFromMemRef(int64_t rank, void *ptr) {
  ::UnrankedMemRefType<int8_t> memref{rank, ptr};
  DynamicMemRefType<int8_t> dynamic(memref);
  std::string result;
  int64_t count = 0;
  if (dynamic.rank > 0 && dynamic.sizes)
    count = dynamic.sizes[0];
  result.reserve(static_cast<size_t>(std::max<int64_t>(count, 0)));
  for (int64_t i = 0; i < count; ++i)
    result.push_back(static_cast<char>(dynamic.data[dynamic.offset + i]));
  return result;
}
void printDynamicCounterSummaries() {
  std::vector<int64_t> counterIds;
  counterIds.reserve(gDynamicCounters.size());
  for (const auto &it : gDynamicCounters)
    counterIds.push_back(it.first);

  std::sort(counterIds.begin(), counterIds.end());
  if (!counterIds.empty()) {
    std::cout << "\n--- Dynamic Counter Totals ---" << std::endl;
    for (int64_t counterId : counterIds) {
      std::cout << "SODA_DYNAMIC_COUNTER name=" << dynamicCounterName(counterId)
                << "\tcount=" << gDynamicCounters[counterId] << std::endl;
    }
  }
}

void printFunctionCounterSummaries() {
  std::vector<std::string> functionNames;
  functionNames.reserve(gFunctionDynamicCounters.size());
  for (const auto &it : gFunctionDynamicCounters)
    functionNames.push_back(it.first);

  if (functionNames.empty())
    return;

  std::sort(functionNames.begin(), functionNames.end());
  std::cout << "\n--- Dynamic Counter Totals By Function (SDCF=SODA_DYNAMIC_COUNTER_FUNCTION) ---" << std::endl;
  for (const std::string &functionName : functionNames) {
    const auto &counterMap = gFunctionDynamicCounters[functionName];
    if (counterMap.empty())
      continue;

    std::vector<int64_t> counterIds;
    counterIds.reserve(counterMap.size());
    for (const auto &it : counterMap)
      counterIds.push_back(it.first);
    std::sort(counterIds.begin(), counterIds.end());

    for (int64_t counterId : counterIds) {
      std::cout << "SDCF function=" << functionName
                << "\tname=" << dynamicCounterName(counterId)
                << "\tcount=" << counterMap.at(counterId) << std::endl;
    }
    std::cout << std::endl;
  }
}

void printGroupCounterSummaries(int64_t groupId) {
  std::vector<int64_t> counterIds;
  for (const auto &it : gGroupDynamicCounters) {
    if (it.first >> 32 == groupId)
      counterIds.push_back(it.first & 0xFFFFFFFF);
  }

  if (!counterIds.empty()) {
    auto nameIt = gGroupFunctionNames.find(groupId);
    if (nameIt != gGroupFunctionNames.end() && !nameIt->second.empty()) {
      std::cout << "\n---  Loop Group " << groupId << " (func "
                << nameIt->second << ") ---" << std::endl;
    } else {
      std::cout << "\n--- Loop Group " << groupId << " ---" << std::endl;
    }
    std::sort(counterIds.begin(), counterIds.end());
    for (int64_t counterId : counterIds) {
      int64_t key = (groupId << 32) | counterId;
      std::cout << "SODA_DYNAMIC_COUNTER name=" << dynamicCounterName(counterId)
                << "\tcount=" << gGroupDynamicCounters[key] << std::endl;
    }
  }
}

void printAllCounterSummaries() {
  printAllLoopSummaries();
  printFunctionCounterSummaries();
  printDynamicCounterSummaries();
}
} // namespace

extern "C" void pre_issue_APE_request() {}

// ---------- addr_gen2.py C implementations ----------

extern "C" void ape_incomplete() {
  std::cout << "APE_INCOMPLETE: unsupported linalg op" << std::endl;
}

extern "C" void ape_broadcast(int64_t aligned, int64_t offset,
                               int64_t s0, int64_t s1,
                               int64_t str0, int64_t str1,
                               int64_t elem_bytes) {
  int64_t eb = clampPositive(elem_bytes);
  for (int64_t idx = 0; idx < s0 * s1; ++idx) {
    int64_t i = idx / (s1 > 0 ? s1 : 1);
    int64_t j = idx % (s1 > 0 ? s1 : 1);
    uint64_t addr = static_cast<uint64_t>(aligned + (offset + i * str0 + j * str1) * eb);
    std::cout << "APE_BROADCAST[" << idx << "]: 0x" << std::hex << addr
              << std::dec << std::endl;
  }
}

extern "C" void ape_gather_from_memref(
    int64_t a_al, int64_t a_off, int64_t a_s0, int64_t a_s1,
    int64_t a_str0, int64_t a_str1,
    int64_t b_al, int64_t b_off, int64_t b_s0, int64_t b_s1,
    int64_t b_str0, int64_t b_str1,
    int64_t c_al, int64_t c_off, int64_t c_s0, int64_t c_s1,
    int64_t c_str0, int64_t c_str1,
    int64_t L1_cache_size, int64_t elem_bytes) {
  int64_t M  = a_s0;
  int64_t K  = a_s1;
  int64_t N  = c_s1;
  int64_t eb = clampPositive(elem_bytes);
  int64_t total = M * N * (2 * K + 1);
  int64_t steps = ((L1_cache_size / elem_bytes) > 0 && (L1_cache_size / elem_bytes) < total) ? (L1_cache_size / elem_bytes) : total;
  for (int64_t index = 0; index < steps; ++index) {
    int64_t ij    = index / (2 * K + 1);
    int64_t i     = ij / N;
    int64_t j     = ij % N;
    int64_t step  = index % (2 * K + 1);
    int64_t k     = step / 2;
    int64_t r     = step % 2;
    int64_t is_c  = (step == 2 * K) ? 1 : 0;
    int64_t A_addr = a_al + (a_off + i * a_str0 + k * a_str1) * eb;
    int64_t B_addr = b_al + (b_off + k * b_str0 + j * b_str1) * eb;
    int64_t C_addr = c_al + (c_off + i * c_str0 + j * c_str1) * eb;
    int64_t addr = (1 - is_c) * ((r == 0) * A_addr + (r == 1) * B_addr)
                 + is_c * C_addr;
    std::cout << "APE_GATHER[" << index << "]: 0x" << std::hex
              << static_cast<uint64_t>(addr) << std::dec << std::endl;
  }
}

// desc_ptr → flat buffer: n_operands × [aligned, offset, s0, s1, str0, str1]
extern "C" void ape_elementwise_trace(int64_t n_operands, int64_t L1_cache_size,
                                       int64_t elem_bytes, int64_t desc_ptr) {
  if (n_operands <= 0 || desc_ptr == 0)
    return;
  const int64_t *descs =
      reinterpret_cast<const int64_t *>(static_cast<uintptr_t>(desc_ptr));
  constexpr int64_t kFields = 6; // aligned, offset, s0, s1, str0, str1
  int64_t out_s0 = descs[(n_operands - 1) * kFields + 2];
  int64_t out_s1 = descs[(n_operands - 1) * kFields + 3];
  if (out_s1 <= 0) out_s1 = 1;
  int64_t M  = out_s0;
  int64_t N  = out_s1;
  int64_t eb = clampPositive(elem_bytes);
  int64_t total = M * N * n_operands;
  int64_t steps = ((L1_cache_size/elem_bytes) > 0 && (L1_cache_size/elem_bytes) < total) ? (L1_cache_size/elem_bytes) : total;
  for (int64_t index = 0; index < steps; ++index) {
    int64_t element = index / n_operands;
    int64_t operand = index % n_operands;
    int64_t i       = element / N;
    int64_t j       = element % N;
    int64_t aligned = descs[operand * kFields + 0];
    int64_t off     = descs[operand * kFields + 1];
    int64_t str0    = descs[operand * kFields + 4];
    int64_t str1    = descs[operand * kFields + 5];
    uint64_t addr   = static_cast<uint64_t>(aligned + (off + i * str0 + j * str1) * eb);
    std::cout << "APE_ELEMENTWISE[" << index << "]: 0x" << std::hex << addr
              << std::dec << std::endl;
  }
}

// ----------------------------------------------------

extern "C" void issue_APE_request(int64_t baseAddr, int32_t tensorId,
                                   int32_t strategy, bool isWrite,
                                   int64_t elementBytes,
                                   int64_t totalElements, int64_t rank,
                                   int64_t shapeDescPtr,
                                   int64_t stridesDescPtr) {
  (void)tensorId;
  (void)strategy;
  if (isWrite)
    return;

  int64_t step = kCacheLineSizeElements;
  int64_t elemBytes = clampPositive(elementBytes);
  int64_t elementCount = clampPositive(totalElements);
  uint64_t base = static_cast<uint64_t>(baseAddr);

  int64_t safeRank = rank > 0 ? rank : 1;
  constexpr int64_t kMaxReasonableRank = 16;
  safeRank = std::min<int64_t>(safeRank, kMaxReasonableRank);

  std::cout << "SODA_APE_DESC rank=" << safeRank
            << " shape="
            << formatI64Descriptor(static_cast<uint64_t>(shapeDescPtr),
                                   safeRank)
            << " strides="
            << formatI64Descriptor(static_cast<uint64_t>(stridesDescPtr),
                                   safeRank)
            << std::endl;

  printAddrLinear(base, elementCount, step, elemBytes);
}

extern "C" void sodaInstrCollectOpCounts(int64_t run, int64_t loopId,
                                          int64_t loads, int64_t stores,
                                          int64_t fpArith,
                                          int64_t intArith) {
  if (!gAtExitRegistered) {
    std::atexit(printAllCounterSummaries);
    gAtExitRegistered = true;
  }

  if (run != 0) {
    gCounterByLoopId.try_emplace(loopId);
    gActiveLoopStack.push_back(loopId);
    return;
  }

  auto stateIt = gCounterByLoopId.find(loopId);
  if (stateIt == gCounterByLoopId.end()) {
    gCounterByLoopId.try_emplace(loopId);
    stateIt = gCounterByLoopId.find(loopId);
  }

  // Attribute this event to all active loops so outer loops include work done
  // by nested loops while they are active.
  for (int64_t activeLoopId : gActiveLoopStack) {
    gCounterByLoopId[activeLoopId].add(loads, stores, fpArith, intArith);
  }

  
  auto activeIt = std::find(gActiveLoopStack.rbegin(), gActiveLoopStack.rend(),
                            loopId);
  if (activeIt != gActiveLoopStack.rend()) {
    gActiveLoopStack.erase(std::next(activeIt).base());
  }
}

extern "C" void sodaInstrDynamicCounter(int64_t counterId, int64_t delta) {
  if (!gAtExitRegistered) {
    std::atexit(printAllCounterSummaries);
    gAtExitRegistered = true;
  }

  // Track counter in global map
  gDynamicCounters[counterId] += delta;
  
  // If we're in a group, also track group-specific counters
  if (gCurrentGroupId >= 0) {
    int64_t key = (gCurrentGroupId << 32) | counterId;
    gGroupDynamicCounters[key] += delta;
  }
}

extern "C" void sodaInstrDynamicCounterStartGroup(int64_t groupId) {
  gCurrentGroupId = groupId;
}

extern "C" void sodaInstrDynamicCounterSetGroupFunctionName(
    int64_t groupId, int64_t charCode) {
  if (charCode < 0 || charCode > 127)
    return;
  gGroupFunctionNames[groupId].push_back(static_cast<char>(charCode));
}

extern "C" void sodaInstrDynamicCounterFlush(int64_t groupId) {
  // Print group-specific counters.
  printGroupCounterSummaries(groupId);

  // Accumulate loop-group counters into per-function totals.
  std::string functionName = "unknown";
  auto nameIt = gGroupFunctionNames.find(groupId);
  if (nameIt != gGroupFunctionNames.end() && !nameIt->second.empty())
    functionName = nameIt->second;
  for (const auto &it : gGroupDynamicCounters) {
    if (it.first >> 32 == groupId) {
      int64_t counterId = it.first & 0xFFFFFFFF;
      gFunctionDynamicCounters[functionName][counterId] += it.second;
    }
  }

  // Disable grouping after the loop flush.
  if (gCurrentGroupId == groupId)
    gCurrentGroupId = -1;

  // Reset group counters for this group.
  std::vector<int64_t> keysToErase;
  for (auto &it : gGroupDynamicCounters) {
    if (it.first >> 32 == groupId) {
      keysToErase.push_back(it.first);
    }
  }
  for (int64_t key : keysToErase) {
    gGroupDynamicCounters.erase(key);
  }

  gGroupFunctionNames.erase(groupId);
}

extern "C" void sodaInstrMarkMatrixAccessStarts(int64_t baseAddr) {
  std::cout << "beginning matrix access" << std::endl;
  std::cout << "SODA_MATRIX_BASE addr=0x" << std::hex
            << static_cast<uint64_t>(baseAddr) << std::dec << std::endl;
}

extern "C" void sodaInstrTraceLinalg(int64_t opId, int64_t opNameRank,
                                      void *opNamePtr, int64_t numInputs,
                                      int64_t numOutputs, int64_t mapsRank,
                                      void *mapsPtr, int64_t itersRank,
                                      void *itersPtr) {
  std::string opName = stringFromMemRef(opNameRank, opNamePtr);
  std::string maps = stringFromMemRef(mapsRank, mapsPtr);
  std::string iteratorTypes = stringFromMemRef(itersRank, itersPtr);

  std::cout << "SODA_LINALG " << opName << ": ins=" << numInputs
            << " outs=" << numOutputs;
  if (maps != "-")
    std::cout << " maps=" << maps;
  if (iteratorTypes != "-")
    std::cout << " iterator_types=" << iteratorTypes;
  std::cout << std::endl;
}

extern "C" void sodaInstrTraceLinalgMemref(
    int64_t opId, int64_t operandKind, int64_t operandIndex, int64_t baseAddr,
    int64_t offset, int64_t dimsRank, void *dimsPtr, int64_t stridesRank,
    void *stridesPtr) {
  (void)opId;
  ::UnrankedMemRefType<int64_t> dimsMemref{dimsRank, dimsPtr};
  ::UnrankedMemRefType<int64_t> stridesMemref{stridesRank, stridesPtr};
  DynamicMemRefType<int64_t> dims(dimsMemref);
  DynamicMemRefType<int64_t> strides(stridesMemref);

  std::cout << "  " << linalgOperandKindName(operandKind) << '['
            << operandIndex << "]"
            << " base=0x" << std::hex << static_cast<uint64_t>(baseAddr)
            << std::dec << " offset=" << offset
            << " dims=" << formatVector(dims)
            << " strides=" << formatVector(strides) << std::endl;
}
