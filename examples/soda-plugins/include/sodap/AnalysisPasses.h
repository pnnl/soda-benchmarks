// Extracted from:
// https://github.com/pnnl/soda-opt/blob/main/include/soda/Dialect/Linalg/Reports/Passes.h
// Under BSD2 License

//===- Passes.h - Linalg Reports pass entry points --------------*- C++ -*-===//

#ifndef SODAP_ANALYSIS_PASSES_H
#define SODAP_ANALYSIS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <memory>
#include <numeric>

namespace mlir {
class Pass;

namespace sodap {
namespace linalg {
namespace reports {

#define GEN_PASS_DECL
#include "sodap/AnalysisPasses.h.inc"
// Struct to hold linalg operation info.
// This info includes:
// - operation name
// - number of inputs
// - number of outputs
// - size of each input/output
// - data type of each input/output
// - bitwidth of each input/output
// - direction of each input/output
// - number of arithmetic instructions inside the kernel
// - number of memory instructions inside the kernel
// - given known sizes:
//   - estimative on total number of arithmetic operations
//   - estimative on total number of memory operations
struct LinalgOpInfo {
  std::string opName;
  int numInputs = 0;
  int numOutputs = 0;
  std::vector<int> inputSizes;
  std::vector<int> outputSizes;
  std::vector<std::string> inputTypes;
  std::vector<std::string> outputTypes;
  std::vector<int> inputTypesBitwidth;
  std::vector<int> outputTypesBitwidth;
  std::vector<std::string> inputDirections;
  std::vector<std::string> outputDirections;
  int numArithmeticOpsInKernel = 0;
  int numMemoryOpsInKernel = 0;
  int numArithmeticOpsEstimative = 0;
  int numMemoryOpsEstimative = 0;

  void printInfo() const {
    llvm::outs() << "Linalg Operation Info:\n";
    llvm::outs() << "Operation Name: " << opName << "\n";
    llvm::outs() << "Number of Inputs: " << numInputs << "\n";
    llvm::outs() << "Number of Outputs: " << numOutputs << "\n";
    llvm::outs() << "Input Sizes: ";
    for (const auto &size : inputSizes) {
      llvm::outs() << size << " ";
    }
    llvm::outs() << "\nOutput Sizes: ";
    for (const auto &size : outputSizes) {
      llvm::outs() << size << " ";
    }
    llvm::outs() << "\nInput Types: ";
    for (const auto &type : inputTypes) {
      llvm::outs() << type << " ";
    }
    llvm::outs() << "\nOutput Types: ";
    for (const auto &type : outputTypes) {
      llvm::outs() << type << " ";
    }
    llvm::outs() << "\nInput Types Bitwidth: ";
    for (const auto &bitwidth : inputTypesBitwidth) {
      llvm::outs() << bitwidth << " ";
    }
    llvm::outs() << "\nOutput Types Bitwidth: ";
    for (const auto &bitwidth : outputTypesBitwidth) {
      llvm::outs() << bitwidth << " ";
    }
    llvm::outs() << "\nInput Directions: ";
    for (const auto &direction : inputDirections) {
      llvm::outs() << direction << " ";
    }
    llvm::outs() << "\nOutput Directions: ";
    for (const auto &direction : outputDirections) {
      llvm::outs() << direction << " ";
    }
    llvm::outs() << "\nNumber of Arithmetic Ops in Kernel: "
                 << numArithmeticOpsInKernel
                 << "\nNumber of Memory Ops in Kernel: "
                 << numMemoryOpsInKernel
                 << "\nEstimated Number of Arithmetic Ops: "
                 << numArithmeticOpsEstimative
                 << "\nEstimated Number of Memory Ops: "
                 << numMemoryOpsEstimative
                 << "\n";
  }
};

// Make getting the size of a memref shape reusable
static int getSizeOfMemRefShape(MemRefType type) {
  auto shape = type.getShape();
  int size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

static int getSizeOfMemRefShape(TensorType type) {
  auto shape = type.getShape();
  int size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

static void getInputSizes(mlir::linalg::GenericOp op, std::vector<int> &sizes) {
  for (auto x : op.getInputs()) {
    Type type = x.getType();
    if (MemRefType mr = dyn_cast<MemRefType>(type))
      sizes.push_back(getSizeOfMemRefShape(
          mr)); // can likely use MemRefType.getNumElements()
    else if (TensorType t = dyn_cast<TensorType>(type))
      sizes.push_back(getSizeOfMemRefShape(t));
    else
      sizes.push_back(0);
  }
}

static void getOutputSizes(mlir::linalg::GenericOp op,
                           std::vector<int> &sizes) {
  for (auto x : op.getOutputs()) {
    Type type = x.getType();
    if (MemRefType mr = dyn_cast<MemRefType>(type))
      sizes.push_back(getSizeOfMemRefShape(mr));
    else if (TensorType t = dyn_cast<TensorType>(type))
      sizes.push_back(getSizeOfMemRefShape(t));
    else
      sizes.push_back(0);
  }
}

static void pushTypeToVector(Type type, std::vector<std::string> &types) {
  // Type does not have a str() method.
  // Thus we print the type to a stream,
  // then convert the stream to a string.
  std::string strType;
  llvm::raw_string_ostream rso(strType);
  type.print(rso);
  types.push_back(rso.str());
}

static void getInputElementType(mlir::linalg::GenericOp op,
                                std::vector<std::string> &types) {
  for (auto x : op.getInputs()) {
    pushTypeToVector(x.getType(), types);
  }
}

static void getOuputElementType(mlir::linalg::GenericOp op,
                                std::vector<std::string> &types) {
  for (auto x : op.getOutputs()) {
    pushTypeToVector(x.getType(), types);
  }
}

static void getInputElementTypeBitwidth(mlir::linalg::GenericOp op,
                                        std::vector<int> &bitwidths) {
  for (auto x : op.getInputs()) {
    Type type = x.getType();
    if (MemRefType mr = dyn_cast<MemRefType>(type))
      bitwidths.push_back(mr.getElementTypeBitWidth());
    else if (TensorType t = dyn_cast<TensorType>(type))
      bitwidths.push_back(t.getElementTypeBitWidth());
    else
      bitwidths.push_back(0);
  }
}

static void getOuputElementTypeBitwidth(mlir::linalg::GenericOp op,
                                        std::vector<int> &bitwidths) {
  for (auto x : op.getOutputs()) {
    Type type = x.getType();
    if (MemRefType mr = dyn_cast<MemRefType>(type))
      bitwidths.push_back(mr.getElementTypeBitWidth());
    else if (TensorType t = dyn_cast<TensorType>(type))
      bitwidths.push_back(t.getElementTypeBitWidth());
    else
      bitwidths.push_back(0);
  }
}

static void getNumArithmeticOpsInKernel(mlir::linalg::GenericOp op,
                                        int &numArithmeticOps) {
                                      
  // Implement a walk over the inner kernel of a linalg.generic op
  // and count the number of arithmetic operations
  numArithmeticOps = 0;
  for (auto &x : /*bb*/ op.getBody()->getOperations()) {
    // Get the operation name and if contains `arith` then increment the counter
    if (x.getName().getStringRef().contains("arith")) {
      numArithmeticOps++;
    }
  }
}

// Compute a simple bound (min = 0, max = size-1) for affine exprs
static std::pair<int64_t, int64_t>
getAffineExprBound(AffineExpr expr, const Value &v) {
  auto shapedTy = dyn_cast<ShapedType>(v.getType());

  if (!shapedTy || !shapedTy.hasRank()) {
    llvm::outs() << "Value type is not a ranked ShapedType.\n";
    llvm::outs() << "Value type: ";
    v.getType().print(llvm::outs());
    llvm::outs() << "\n";
    return {0, 0};
  }

  // Case 1: Dimension expression
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    llvm::outs() << "Processing AffineDimExpr\n";
    dimExpr.print(llvm::outs());
    llvm::outs() << "\n";
    unsigned pos = dimExpr.getPosition();
    if (pos >= shapedTy.getRank()) {
      llvm::outs() << "Warning: dim position " << pos
                   << " exceeds type rank " << shapedTy.getRank() << "\n";
      return {0, 0};
    }

    int64_t dimSize = shapedTy.getDimSize(pos);
    if (dimSize == ShapedType::kDynamic) {
      // Unknown bound — conservatively set to 0
      llvm::outs() << "Dynamic dim size for dim " << pos << "\n";
      return {0, 0};
    }

    llvm::outs() << "Dim position: " << pos << ", size: " << dimSize << "\n";
    return {0, dimSize - 1};
  }

  // Case 2: Constant
  if (auto cstExpr = dyn_cast<AffineConstantExpr>(expr)) {
    llvm::outs() << "Processing AffineConstantExpr\n";
    int64_t val = cstExpr.getValue();
    llvm::outs() << "Constant value: " << val << "\n";
    return {val, val};
  }

  // Case 3: Binary op (recursive)
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
    llvm::outs() << "Processing AffineBinaryOpExpr\n";
    llvm::outs() << "LHS: ";
    binExpr.getLHS().print(llvm::outs());
    llvm::outs() << "\nRHS: ";
    binExpr.getRHS().print(llvm::outs());
    llvm::outs() << "\n";
    auto lhsBound = getAffineExprBound(binExpr.getLHS(), v);
    auto rhsBound = getAffineExprBound(binExpr.getRHS(), v);

    switch (binExpr.getKind()) {
    case AffineExprKind::Add:
      return {lhsBound.first + rhsBound.first,
              lhsBound.second + rhsBound.second};

    // case AffineExprKind::Sub:
    //   return {lhsBound.first - rhsBound.second,
    //           lhsBound.second - rhsBound.first};

    case AffineExprKind::Mul: {
      int64_t a = lhsBound.first * rhsBound.first;
      int64_t b = lhsBound.first * rhsBound.second;
      int64_t c = lhsBound.second * rhsBound.first;
      int64_t d = lhsBound.second * rhsBound.second;
      return {std::min({a, b, c, d}), std::max({a, b, c, d})};
    }

    default:
      llvm::outs() << "Unsupported binary op in affine expr\n";
      return {0, 0};
    }
  }

  llvm::outs() << "Unhandled affine expression kind\n";
  return {0, 0};
}

static void getNumMemoryOpsInKernel(mlir::linalg::GenericOp op,
                                    int &numMemoryOps) {
  // To account for the memory operations we must check which basic block
  // arguments are used inside the kernel If a bba is used, it means that a load
  // operation is being performed If there is yielded values in linalg.yield, it
  // means that a store operation is being performed
  int numLoadOps = 0;
  int numStoreOps = 0;
  for (auto &bba : op.getBody()->getArguments()) {
    if (bba.use_empty())
      continue;
    numLoadOps++;
  }

  auto yieldOp = op.getBody()->getTerminator();
  assert(isa<mlir::linalg::YieldOp>(yieldOp));
  numStoreOps = yieldOp->getNumOperands();

  numMemoryOps = numLoadOps + numStoreOps;
}

static int getNumberOfIterations(mlir::linalg::GenericOp op) {
  // To provide an estimative on the number of arithmetic operations,
  // we can use the number of loops and the loop bounds for this GenericOp.

  // The number of loops derives from the number of iterator types (e.g.
  // parallel, reduction, etc.)
  // int numLoops = op.getIteratorTypes().size();

  // The bounds of the loops can be obtained from the `indexing_maps` attribute
  // and the dimensions of the input and output tensors Each genericOp has a
  // list of indexing maps (affine_map), one for each input and output tensor
  // ex: #map = affine_map<(d0, d1, d2) -> (d0, d2)>
  // In the example the input/output that uses this map has 2 dimensions and the
  // loop bounds 0 and 2 are d0 and d2
  // We will reduce with the product of the loop bounds
  std::vector<int> loopBounds;

  // All maps have the same number of dimensions, so we can use the first one to
  // get the number of dimensions
  auto mapAttr = op.getIndexingMaps()[0];
  auto map = cast<AffineMapAttr>(mapAttr).getValue();
  int numDims = map.getNumDims();

  // Now we create a dictionary to store if we have already processed a
  // dimension so we don't count dimensions twice
  std::map<int, bool> processedDims;
  for (int i = 0; i < numDims; i++) {
    processedDims[i] = false;
  }

  // Iterate over all indexing_map, input/output pairs.
  int count = 0;
  int numInputs = op.getInputs().size();
  int numOutputs = op.getOutputs().size();
  for (auto &mapAttr : op.getIndexingMaps()) {
    // Assert count is in range of inputs and outputs.
    assert(count < numInputs + numOutputs &&
           "Number of indexing_maps does not match number of inputs and "
           "outputs.");

    // Inputs and outputs are split in two different ranges.
    Value v;
    if (count < numInputs) {
      v = op.getInputs()[count];
    } else {
      v = op.getOutputs()[count - numInputs];
    }
    count++;

    auto map = cast<AffineMapAttr>(mapAttr).getValue();

    // for (unsigned i = 0; i < map.getNumResults(); i++) {
    //   if (processedDims[map.getDimPosition(i)] == false) {
    //     processedDims[map.getDimPosition(i)] = true;
    //     loopBounds.push_back(cast<ShapedType>(v.getType()).getDimSize(i));
    //   } else {
    //     // If we have already processed this dimension, we can skip it
    //     continue;
    //   }
    // }

    llvm::outs() << "Processing indexing map for value: ";
    v.getType().print(llvm::outs());
    llvm::outs() << "\n";
     
    llvm::outs() << "Affine Map: ";
    map.print(llvm::outs());
    llvm::outs() << "\n";

    llvm::outs() << "Num Results in Map: " << map.getNumResults() << "\n";

    for (unsigned i = 0; i < map.getNumResults(); ++i) {
      // Get the result affine expression
      AffineExpr result = map.getResult(i);
      llvm::outs() << "Result Expr " << i << ": ";
      result.print(llvm::outs());
      llvm::outs() << "\n";
      auto [lower, upper] = getAffineExprBound(result, v);
      loopBounds.push_back(upper);
    }

    // ---------------------------
    // Print debug information
    // llvm::outs() << "\n";
    // map.print(llvm::outs());
    // llvm::outs() << "\n";
    // llvm::outs() << "NumDims: " << map.getNumDims() << "\n";
    // llvm::outs() << "NumSymbols: " << map.getNumSymbols() << "\n";
    // llvm::outs() << "NumResults: " << map.getNumResults() << "\n";
    // llvm::outs() << "NumInputs: " << map.getNumInputs() << "\n";
    // llvm::outs() << "\n";
    // // Print results
    // for (int i = 0; i < map.getNumResults(); i++) {
    //   llvm::outs() << "Result " << i << ": " << map.getResult(i) << "\n";
    // }
    // // Print DimPosition
    // for (int i = 0; i < map.getNumResults(); i++) {
    //   llvm::outs() << "DimPos " << i << ": " << map.getDimPosition(i) <<
    //   "\n";
    // }
    // // Print input.getType()
    // llvm::outs() << "Argument Type: " << v.getType() << "\n";
    // llvm::outs() << "\n";
    // ---------------------------

    // Check if we processed all dimensions so we can early exit the loop
    bool allProcessed = true;
    for (int i = 0; i < numDims; i++) {
      if (processedDims[i] == false) {
        allProcessed = false;
        break;
      }
    }
    if (allProcessed) {
      break;
    }
  }

  // ---------------------------
  // Print debug information
  // llvm::outs() << "\n";
  // llvm::outs() << "Loop Bounds: ";
  // for (auto &bound : loopBounds) {
  //   llvm::outs() << bound << " ";
  // }
  // llvm::outs() << "\n";
  // ---------------------------

  // Now we can calculate the number of iterations
  int numberOfIterations = 1;
  for (auto &bound : loopBounds) {
    numberOfIterations *= bound;
  }

  return numberOfIterations;
}

static int64_t computeLinalgIterations(mlir::linalg::LinalgOp linalgOp) {
  llvm::SmallVector<int64_t> ranges = linalgOp.getStaticLoopRanges();
  // If we encounter any dynamic ranges, try to get runtime value or use a default
  for (auto &r : ranges) {
    if (r == ShapedType::kDynamic) {
      // For now, conservatively set dynamic sizes to 1
      // TODO: Could try to:
      // 1. Get runtime value if available
      // 2. Use a heuristic default size
      // 3. Extract from dynamic dimension ops
      r = 1;
      // llvm::outs() << "Warning: Dynamic range encountered, using 1 as conservative estimate\n";
    }
  }
  return std::accumulate(ranges.begin(), ranges.end(), 1LL,
                         std::multiplies<int64_t>());
}

static void getNumArithmeticOpsEstimative(mlir::linalg::GenericOp op,
                                          LinalgOpInfo &opInfo) {
  int numOfIterations = computeLinalgIterations(op);
  // int numOfIterations = getNumberOfIterations(op);
  
  opInfo.numArithmeticOpsEstimative =
      numOfIterations * opInfo.numArithmeticOpsInKernel;
}

static void getNumMemoryOpsEstimative(mlir::linalg::GenericOp op,
                                      LinalgOpInfo &opInfo) {
  // int numOfIterations = getNumberOfIterations(op);
  int numOfIterations = computeLinalgIterations(op);

  opInfo.numMemoryOpsEstimative = numOfIterations * opInfo.numMemoryOpsInKernel;
}

static void collectLinalgOperationInfo(LinalgOpInfo &opInfo,
                                       mlir::linalg::GenericOp op) {
  // std::string kernelFnName =
      // opInfo.opName =
      // op->getParentOfType<func::FuncOp>().getName()+"/"+op->getName();
      opInfo.opName = Twine(op->getParentOfType<func::FuncOp>().getName(),
                            "/linalg.generic")
                          .str();
  opInfo.numInputs = op.getInputs().size();
  opInfo.numOutputs = op.getOutputs().size();
  getInputSizes(op, opInfo.inputSizes);
  getOutputSizes(op, opInfo.outputSizes);
  getInputElementType(op, opInfo.inputTypes);
  getOuputElementType(op, opInfo.outputTypes);
  getInputElementTypeBitwidth(op, opInfo.inputTypesBitwidth);
  getOuputElementTypeBitwidth(op, opInfo.outputTypesBitwidth);
  // getInputDirections();
  // getOutputDirections();
  getNumArithmeticOpsInKernel(op, opInfo.numArithmeticOpsInKernel);
  getNumMemoryOpsInKernel(op, opInfo.numMemoryOpsInKernel);
  getNumArithmeticOpsEstimative(op, opInfo);
  getNumMemoryOpsEstimative(op, opInfo);
}

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
