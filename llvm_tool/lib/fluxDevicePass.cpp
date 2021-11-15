#include "Mekong-Utils.h"
#include "cudaFluxPasses.h"
#include "deviceRuntime.h"
#include "ptx_parser.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <array>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llvm;

std::string exec(const char *cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
  if (!pipe)
    throw std::runtime_error("popen() failed!");
  while (!feof(pipe.get())) {
    if (fgets(buffer.data(), 128, pipe.get()) != nullptr)
      result += buffer.data();
  }
  return result;
}

std::vector<mekong::PTXFunction> ptxInstructionAnalysis(Module &M) {
  // Get Module Prefix
  std::string prefix = mekong::getModulePrefix(&M);

  errs() << "CUDA Flux: Module prefix: " << prefix << "\n";

  // Write IR to file
  std::string byteCodeFile = prefix + ".bc";
  mekong::dumpModuleToFile(M, byteCodeFile);

  // call llc to get ptx code
  // get target processor and target features
  std::string ptxFile = prefix + ".ptx";
  std::vector<Function *> kernels;
  mekong::getKernels(M, kernels);
  StringRef target_proc;
  StringRef target_features;
  for (Function *kernel : kernels) {
    target_proc = kernel->getFnAttribute("target-cpu").getValueAsString();
    target_features = kernel->getFnAttribute("target-features").getValueAsString();
    break; // assume all kernel have the same attributes
  }

  // use O2 because at this point all the higher optimizations are already done.
  // O2 ensures that the kernel is not simplyfied (de-optimized) again
  std::string llc_cmd =
      "llc -O2 -mcpu=" + target_proc.str() + " -mattr=" + target_features.str() + " -o " + ptxFile + " " + byteCodeFile;
  exec(llc_cmd.c_str());

  // parse ptx and write to file
  std::string outputFile = prefix + ".out";
  auto tokenVec = mekong::lexicalAnalysis(ptxFile);
  auto funcVec = mekong::parse(tokenVec);

  // Write PTX Analysis to File
  // do this befor asserting congruent basic blocks
  // for easier debugging
  {
    std::ofstream outfileM(outputFile, std::ios::binary);

    for (auto func : funcVec) {
      // Making out file for every kernel and writing its ptx instructions to it
      std::string outputFile = func.name + ".out";
      std::ofstream outfile(outputFile, std::ios::binary);
      outfileM << "- Function:\n";
      outfileM << "  Name: " << func.name << "\n";
      outfileM << "  BlockCount: " << func.bb.size() << "\n";
      outfileM << "  Blocks: \n";
      for (auto bb : func.bb) {
        outfile << "--\n";
        outfileM << "  - " << bb.name << ":\n";
        for (auto inst : bb.inst) {
          outfile << inst << "\n";
          outfileM << "    - " << inst << "\n";
        }
      }
    }
  }

  // Get Register Usage
  {
    // errs() << "Parsing Register Usage...\n";
    // TODO path will not work if not already installed. this will fail when
    // testing std::string python_cmd = "parseRegisterUsage.py " + ptxFile + " "
    // + outputFile; exec(python_cmd.c_str());
  }

  return funcVec;
}

void basicBlockInstrumentation(Module &M, std::vector<mekong::PTXFunction> funcVec) {

  LLVMContext &ctx = M.getContext();

  // Get Trace Function
  Function *traceFun = M.getFunction("incBlockCounter_mt_2");

  // GetKernels //
  std::vector<Function *> kernels;
  mekong::getKernels(M, kernels);

  // For each Kernel
  for (Function *kernel : kernels) {
    errs() << "CUDA Flux: Working on kernel: " + kernel->getName() + "\n";
    // TODO Check for function calls
    bool callsFunctions = false;

    // Warn if function calls are made
    if (callsFunctions)
      errs() << "CUDA Flux: WARNING for kernel " + kernel->getName() + ": FunctionCalls are not supported yet!\n";

    // Clone Kernel and add Argument : (AMMAR: Adding our counter)
    std::vector<Type *> args;
    args.push_back(Type::getInt64PtrTy(ctx));
    args.push_back(Type::getInt64PtrTy(ctx));
    args.push_back(Type::getInt32Ty(ctx));
    args.push_back(Type::getInt32Ty(ctx));
    Function *kernelClone = mekong::cloneAndAddArgs(kernel, args, {"bblist", "bblist1", "n_banks", "profiling_mode"});

    // Assign each block an id
    // Must be done on the kernel clone otherwise the mapping is of when
    // instrumenting the basic blocks
    std::map<BasicBlock *, int> blockIDs = mekong::getBlockIDMap(kernelClone, funcVec, kernel->getName().str());
    int block_count = 0;
    for (llvm::BasicBlock &bb : kernelClone->getBasicBlockList()) {
      block_count += 1;
    }
    errs() << "CUDA Flux: BlockCount: " << block_count << "\n";

    // Get TracePointer (2 before last kernel argument)
    Value *trace_ptr = &*(kernelClone->arg_begin() + (kernelClone->arg_size() - 4));
    Value *trace_ptr1 = &*(kernelClone->arg_begin() + (kernelClone->arg_size() - 3));
    // Get Number of Banks (before kernel argument)
    Value *n_banks = &*(kernelClone->arg_begin() + (kernelClone->arg_size() - 2));
    // Get Profiling Mode (last kernel argument)
    Value *profiling = &*(kernelClone->arg_begin() + (kernelClone->arg_size() - 1));

    // Get IRBuilder
    IRBuilder<> builder(ctx);

    Instruction *insertPoint = &*kernelClone->begin()->getFirstInsertionPt();
    insertPoint = insertPoint->getNextNode();
    builder.SetInsertPoint(insertPoint);

    // For each BasicBlock
    for (BasicBlock &bb : kernelClone->getBasicBlockList()) {
      builder.SetInsertPoint(&*bb.getFirstInsertionPt());
      Constant *block_ID = builder.getInt32(blockIDs.at(&bb));

      // errs() << *traceFun << "\n";
      // errs() << *trace_ptr <<"\t"<< *block_ID <<"\t"<< *n_banks << "\n";

      // Increase blockCounter[blockID]
      builder.CreateCall(traceFun, {trace_ptr, trace_ptr1, block_ID, n_banks, profiling});
    }

    // Add metadata so the device function will become a kernel when assembled
    // to a binary
    mekong::markKernel(M, kernelClone);
  }
}

bool FluxDevicePass::runOnModule(Module &M) {
  if (M.getTargetTriple().find("nvptx64") == std::string::npos)
    return false;

  LLVMContext &ctx = M.getContext();
  errs() << "CUDA Flux: Instrumenting device code...\n";

  // Produce PTX Assembly and Analyze it
  auto funcVec = ptxInstructionAnalysis(M);

  // Link Device Runtime //
  // TODO
  // device runtime needs to be compiled for the same gpu arch than the current file
  // Steps:
  // * print gpu arch
  std::vector<Function *> kernels;
  mekong::getKernels(M, kernels);
  StringRef target_proc;
  for (Function *kernel : kernels) {
    target_proc = kernel->getFnAttribute("target-cpu").getValueAsString();
    break; // assume all kernel have the same attributes
  }

  char tmp_file[] = "/tmp/cuda_flux_drtXXXXXX";
  mkstemp(tmp_file);
  std::string rt_path(tmp_file);

  // * create source file in /tmp
  std::error_code EC;
  { // Seperate name space to prevent usage of file later in code
    // also flushes input of file to disk - very important!
    raw_fd_ostream file(rt_path + ".cu", EC, llvm::sys::fs::OpenFlags(1)); // OF_Text
    assert(!EC);
    file.write((const char *)deviceRuntime_cu, deviceRuntime_cu_len);
    file.flush();
  }
  // * compile to byte code
  std::string deviceRuntime_ll;
  exec(std::string("clang++ -v -S -emit-llvm --cuda-device-only --cuda-gpu-arch=" + target_proc.str() +
                   " -O3 -std=c++11 -o " + rt_path + ".ll " + rt_path + ".cu")
           .c_str());
  // * link byte code to current module
  { // Seperate name space to prevent usage of file later in code
    // TODO replace with llvm equivalent
    std::ifstream file(rt_path + ".ll");
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    deviceRuntime_ll = std::string(size, '\0');
    file.seekg(0);
    file.read(&deviceRuntime_ll[0], size);
  }
  StringRef deviceRuntime(deviceRuntime_ll);

  // Link against current module
  mekong::linkIR(deviceRuntime, M);

  // Instrument Each Basic Block with code to count the number of executions
  basicBlockInstrumentation(M, funcVec);

  return true;
}

void FluxDevicePass::getAnalysisUsage(AnalysisUsage &AU) const {
  // AU.setPreservesAll();
}

void FluxDevicePass::releaseMemory() {}
