#include "Mekong-Utils.h"
#include "cudaFluxPasses.h"
#include "hostRuntime.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Transforms/Utils/Cloning.h>

#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace llvm;

bool FluxHostPass::runOnModule(llvm::Module &M) {
  if (M.getTargetTriple().find("x86_64") == std::string::npos)
    return false;

  LLVMContext &ctx = M.getContext();
  errs() << "CUDA Flux: instrumenting host code...\n";

  errs() << "CUDA Flux: CUDA Version ";
  errs() << M.getSDKVersion().getMajor() << "." << M.getSDKVersion().getMinor().getValue() << "\n";

  // Link Device Runtime //
  // Load Memory Buffer from Headerfile
  std::string hostRuntimeNull((const char *)hostRuntime_ll, hostRuntime_ll_len);
  // Add nulltermination
  hostRuntimeNull.append("\0");
  StringRef hostRuntime(hostRuntimeNull.c_str(), hostRuntimeNull.size());
  // Link against current module
  mekong::linkIR(hostRuntime, M);

  // Read BlockAnalysis
  std::string prefix = mekong::getModulePrefix(&M);
  std::string deviceInfoFile = prefix + ".out";
  std::ifstream file(deviceInfoFile, std::ios::binary | std::ios::ate);
  auto size = file.tellg();
  // TODO Fail earlier
  if (size < 0) {
    errs() << "Could not read Device Analysis Info. Assume Hostcode Only. "
              "Terminating...\n";
    return true;
  }
  file.seekg(0, std::ios::beg);

  std::string buffer(size, '\0');
  assert(file.read(&buffer[0], size));

  std::vector<std::string> lines;
  size_t pos = 0;
  size_t pos_end = std::string::npos;
  std::string token;
  while ((pos_end = buffer.find("\n", pos)) != std::string::npos) {
    lines.push_back(buffer.substr(pos, pos_end - pos));
    pos = pos_end + 1;
  }

  // kernel -> BasicBlockCount
  std::map<std::string, int> kernelBlockMap;

  std::string kernelName = "";
  int blockCount = -1;
  for (int i = 0; i < lines.size(); ++i) {
    auto line = lines[i];

    // If new item in yaml list reset
    if (line[0] == '-') {
      kernelName = "";
      blockCount = -1;
    }

    // If blockcount found, parse and store it
    size_t pos = line.find("BlockCount");
    if (pos != std::string::npos) {
      auto count = line.substr(pos + 11); // BlockCount: has 11 characters
      blockCount = std::stoi(count);
    }

    // If name found, also store it
    pos = line.find("Name");
    if (pos != std::string::npos) {
      kernelName = line.substr(pos + 6); // 'Name: ' has 6 characters
    }

    // If name and blockcount have been found a valid entry can be added to the
    // map, reset afterwards
    if (kernelName != "" and blockCount != -1) {
      kernelBlockMap[kernelName] = blockCount;
      kernelName = "";
      blockCount = -1;
    }
  }

  // Insert Global String Containing the BasicBlock Analysis
  // Create Another Global Variable indicating if the ptx analysis was already
  // written to disk

  IRBuilder<> builder(ctx);
  Value *analysisStr = mekong::createGlobalStringPtr(M, StringRef(buffer));
  GlobalVariable *ptxAnalysIsWritten =
      new GlobalVariable(M, builder.getInt8Ty(), false, GlobalValue::PrivateLinkage, builder.getInt8(0),
                         "ptxAnalysisWritten", nullptr, GlobalVariable::NotThreadLocal, 0);

  // Call modified kernel

  // Get Kernel Calls and Kernels
  // Kernel Descriptor:
  // - handle
  // - name
  std::vector<mekong::KernelDescriptor> descs;
  mekong::getKernelDescriptors(M, descs);

  // For Kernel Call
  //   Allocate memory for trace
  //   Memset
  //   Copy Ptr to global variable
  //   Stream Sync
  //   Copy Data Back
  //   Write Log
  //   Free Memory
  for (auto &desc : descs) {

    // Get Name of the Kernel
    std::string kernelName = desc.name.str();
    std::string cloneName = kernelName + "_clone";

    // Add Parameters of original Kernel and add the Array for the BB Counters
    std::vector<Type *> cloneParamenter;
    for (auto *type : desc.handle->getFunctionType()->params())
      cloneParamenter.push_back(type);
    cloneParamenter.push_back(Type::getInt64PtrTy(ctx)); // Pointer to Counters for Basic Blocks
    cloneParamenter.push_back(Type::getInt64PtrTy(ctx)); // Pointer to Counters for Basic Blocks
    cloneParamenter.push_back(Type::getInt32Ty(ctx));    // Number of Banks for the counters
    cloneParamenter.push_back(Type::getInt32Ty(ctx));    // Profiling Mode

    // Create Kernelclone wrapper
    FunctionType *cloneType =
        FunctionType::get(desc.handle->getFunctionType()->getReturnType(), cloneParamenter, false);
    Function *clonedKernelWrapper = mekong::createKernelWrapper(M, cloneName, cloneType);

    mekong::registerKernel(M, cloneName, clonedKernelWrapper);

    std::vector<CallBase *> launchSites;
    mekong::getKernelLaunchSites(desc.handle, launchSites);
    printf("passed before launches loop\n");
    for (auto *launch : launchSites) {
      CallBase *confCall = mekong::getKernelConfigCall(M, launch);
      builder.SetInsertPoint(confCall);

      // Allocate memory for BB Counters

      int64_t blockCount = 100;
      auto result = kernelBlockMap.find(kernelName);
      if (result != kernelBlockMap.end()) {
        blockCount = result->second;
        errs() << "CUDA Flux: Found BasicBlockCount for kernel " << kernelName << ": " << blockCount << "\n";
      } else {
        errs() << "CUDA Flux: BasicBlockCount for kernel " << kernelName << "not found!\n";
      }

      // Set Number of Banks to 64, using the number of SMs would probably be
      // better Optimize later
      int32_t n_banks = 64;
      Value *blockCountVal = builder.getInt64(blockCount * n_banks);

      // Insert function call to create the memory needed for the basic block
      // counters
      Function *createBBCounterMemoryFu = M.getFunction("createBBCounterMemory");
      Value *devPtr = builder.CreateCall(createBBCounterMemoryFu, {blockCountVal});
      Value *n_banks_val = builder.getInt32(n_banks);

      // Allocation of the new counter
      Function *createBBCounterMemory1Fu = M.getFunction("createBBCounterMemory1");
      Value *tracPtr = builder.CreateCall(createBBCounterMemory1Fu, {blockCountVal});
      printf("Passed flux trac pointer creation\n");

      // Insert function call to get the profiling Mode
      Function *getProfilingModeFu = M.getFunction("getProfilingMode");
      Value *profilingMode = builder.CreateCall(getProfilingModeFu);

      // Insert function call to start time measurement
      Function *startClockFu = M.getFunction("startClock");
      Value *timeval = builder.CreateCall(startClockFu);

      // Prepare remaining arguments for the cloned kernel
      std::vector<Value *> additionalArgs;
      additionalArgs.push_back(devPtr);
      // Passing our new counter
      additionalArgs.push_back(tracPtr);
      // printf("passed addition of tracPtr to addArgs\n");
      // Let the kernel know how many banks are used
      additionalArgs.push_back(n_banks_val);
      // Let the kernel know which profiling mode is used
      additionalArgs.push_back(profilingMode);

      printf("passed addition of tracPtr to addArgs\n");

      CallBase *cloneLaunchCall = mekong::replaceKernelLaunch(M, launch, clonedKernelWrapper, additionalArgs);

      printf("Passed cloneLaunchCall\n");
      // Make sure to insert after the launch call
      builder.SetInsertPoint(cloneLaunchCall->getNextNode());

      // Synchronize and Serialize BB Counters
      // ( syncStream, copy to host, write to file)
      Function *serializeCountersFu = M.getFunction("serializeBBCounter");

      Value *kernelStringPtr = builder.CreateGlobalStringPtr(kernelName.c_str());

      // Use 6th Arg of the clone kernel wrapper, which is the stream
      // wrapper( i64 gridXY, i32 gridZ, i64 blockXY, i32 blockZ, int64 shmsize,
      // cudaStream_t stream, ... )
      builder.CreateCall(serializeCountersFu,
                         {devPtr, tracPtr, blockCountVal, n_banks_val, kernelStringPtr,
                          cloneLaunchCall->getArgOperandUse(0), cloneLaunchCall->getArgOperandUse(1),
                          cloneLaunchCall->getArgOperandUse(2), cloneLaunchCall->getArgOperandUse(3),
                          cloneLaunchCall->getArgOperandUse(4), cloneLaunchCall->getArgOperandUse(5), timeval});

      printf("Call creation serializeCountersFu with tracPtr\n");

      // Serializing PTX Analysis
      Function *writePTX_Analysis = M.getFunction("mekong_WritePTXBlockAnalysis");

      builder.CreateCall(writePTX_Analysis, {ptxAnalysIsWritten, analysisStr});
    }
  }

  return true;
}

void FluxHostPass::getAnalysisUsage(AnalysisUsage &AU) const { AU.setPreservesAll(); }

void FluxHostPass::releaseMemory() {}
