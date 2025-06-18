// Extracted from:
// https://github.com/pnnl/soda-opt/blob/main/include/soda/Dialect/Linalg/Reports/Passes.h
// Under BSD2 License

//===- Passes.h - Linalg Reports pass entry points --------------*- C++ -*-===//

#ifndef SODAP_ANALYSIS_PASSES_H
#define SODAP_ANALYSIS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;

namespace sodap {
namespace linalg {
namespace reports {

#define GEN_PASS_DECL
#include "sodap/AnalysisPasses.h.inc"

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//
std::unique_ptr<mlir::Pass> createGenerateLinalgSummary();

//===----------------------------------------------------------------------===//
// Register passes
//===----------------------------------------------------------------------===//
#define GEN_PASS_REGISTRATION
#include "sodap/AnalysisPasses.h.inc"

} // namespace reports
} // namespace linalg
} // namespace sodap
} // namespace mlir

#endif // SODAP_ANALYSIS_PASSES_H
