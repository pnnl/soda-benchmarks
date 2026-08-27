// Extracted from:
// https://github.com/pnnl/soda-opt/blob/main/lib/Dialect/Linalg/Reports/GenerateLinalgSummary.cpp
// Under BSD2 License

//===- GenerateLinalgSummary.cpp - GenerateLinalgSummaryPass --------------===//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that generates a summary of linalg generic
// operations.
//
// See test/Analysis/linalg/operation-info.mlir for summary example.
//
//===----------------------------------------------------------------------===//

#include "sodap/AnalysisPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "test-passes"

namespace mlir::sodap::linalg::reports {
#define GEN_PASS_DEF_GENERATELINALGSUMMARY
#include "sodap/AnalysisPasses.h.inc"
} // namespace mlir::sodap::linalg::reports

using namespace mlir;
using namespace mlir::sodap;

namespace {

class GenerateLinalgSummaryPass
    : public mlir::sodap::linalg::reports::impl::GenerateLinalgSummaryBase<
          GenerateLinalgSummaryPass> {

  void runOnOperation() override {

    getOperation().walk([this](mlir::linalg::GenericOp op) {
      // Prepare the output streams
      std::string errorMessage;
      // std::string filename = op.getKernelName().getValue().str() +
      // "_linalg_summary.txt";
      std::string filename = "linalg_summary.txt";
      auto output = openOutputFile(filename, &errorMessage);
      outputStream = &output->os();

      if (writeToTerminal) {
        outputStream = &llvm::outs();
      }

      // Populate the stream with the xml vector
      resetIndent();

      if (!writeToTerminal) {
        output->keep();
      }

      sodap::linalg::reports::LinalgOpInfo opInfo;
      sodap::linalg::reports::collectLinalgOperationInfo(opInfo, op);
      printLinalgOpInfo(opInfo);
    });
  }

  // To keep track of what name to use for the XML arguments
  int pointerId = 0;

  // Resets pointer ID
  // Should be called at each new testbench
  void resetPointerId() { pointerId = 0; }
  int incPointerId() { return pointerId++; }

  void printAestPreamble() {
    printIndent() << "<?xml version=\"1.0\"?>\n"
                  << "<function>\n";
  }

  void initTestbench() {
    printIndent() << "<testbench\n";
    resetPointerId();
  }

  void closeTestbench() { printIndent() << "/>\n"; }

  void printLinalgOpInfo(const sodap::linalg::reports::LinalgOpInfo &opInfo) {
    printA() << "=========================\n";
    printA() << "REPORT BEGIN\n";
    printA() << "LinalgOpInfo: " << opInfo.opName << "\n";
    printA() << "  numInputs: " << opInfo.numInputs << "\n";
    printA() << "  numOutputs: " << opInfo.numOutputs << "\n";
    printA() << "  inputSizes: ";
    for (auto s : opInfo.inputSizes) {
      printA() << s << " ";
    }
    printA() << "\n";
    printA() << "  outputSizes: ";
    for (auto s : opInfo.outputSizes) {
      printA() << s << " ";
    }
    printA() << "\n";
    printA() << "  inputTypes: ";
    for (auto t : opInfo.inputTypes) {
      printA() << t << " ";
    }
    printA() << "\n";
    printA() << "  outputTypes: ";
    for (auto t : opInfo.outputTypes) {
      printA() << t << " ";
    }
    printA() << "\n";
    printA() << "  inputTypesBitwidth: ";
    for (auto t : opInfo.inputTypesBitwidth) {
      printA() << t << " ";
    }
    printA() << "\n";
    printA() << "  outputTypesBitwidth: ";
    for (auto t : opInfo.outputTypesBitwidth) {
      printA() << t << " ";
    }
    printA() << "\n";
    printA() << "  inputDirections: ";
    for (auto d : opInfo.inputDirections) {
      printA() << d << " ";
    }
    printA() << "\n";
    printA() << "  outputDirections: ";
    for (auto d : opInfo.outputDirections) {
      printA() << d << " ";
    }
    printA() << "\n";
    printA() << "  numArithmeticOpsInKernel: "
             << opInfo.numArithmeticOpsInKernel << "\n";
    printA() << "  numMemoryOpsInKernel: " << opInfo.numMemoryOpsInKernel
             << "\n";
    printA() << "  numArithmeticOpsEstimative: "
             << opInfo.numArithmeticOpsEstimative << "\n";
    printA() << "  numMemoryOpsEstimative: " << opInfo.numMemoryOpsEstimative
             << "\n";
    printA() << "REPORT END\n";
    printA() << "=========================\n";
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IndentRAII {
    int &indent;
    IndentRAII(int &indent) : indent(indent) {}
    ~IndentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  IndentRAII pushIndent() { return IndentRAII(++indent); }

  /// Output streams to the generated XML files or terminal output
  raw_ostream *outputStream;
  raw_ostream &analysisOut() { return *outputStream; }

  llvm::raw_ostream &printIndent() {
    for (int i = 0; i < indent; ++i)
      analysisOut() << " ";
    return analysisOut();
  }

  llvm::raw_ostream &printA() { return analysisOut(); }
};

} // end anonymous namespace

// Generate linalg summary pass
std::unique_ptr<mlir::Pass>
mlir::sodap::linalg::reports::createGenerateLinalgSummary() {
  return std::make_unique<GenerateLinalgSummaryPass>();
}
