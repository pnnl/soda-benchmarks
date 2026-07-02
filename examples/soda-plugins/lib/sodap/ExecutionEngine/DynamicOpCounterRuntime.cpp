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
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>
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
