//===- DynamicOpCounterRuntime.cpp - Runtime support for op counters ------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sodap/ExecutionEngine/DynamicOpCounterRuntime.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "llvm/ADT/ArrayRef.h"
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
std::vector<int64_t> gActiveLoopStack;
bool gAtExitRegistered = false;
int64_t gCurrentGroupId = -1;

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

void printGroupCounterSummaries(int64_t groupId) {
  std::vector<int64_t> counterIds;
  for (const auto &it : gGroupDynamicCounters) {
    if (it.first >> 32 == groupId)
      counterIds.push_back(it.first & 0xFFFFFFFF);
  }

  if (!counterIds.empty()) {
    std::cout << "\n--- Loop Group " << groupId << " ---" << std::endl;
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

extern "C" void sodaInstrDynamicCounterFlush(int64_t groupId) {
  // Print group-specific counters.
  printGroupCounterSummaries(groupId);

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
}
