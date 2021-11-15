#include "cudaFluxPasses.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace llvm;

char FluxDevicePass::ID = 0;
char FluxHostPass::ID = 0;

// Make the pass visible to opt.
static RegisterPass<FluxDevicePass> X("nvptx-cuda-flux", "Instruments basic blocks for profiling");
static RegisterPass<FluxHostPass> Y("host-cuda-flux", "Instrument nvptx kernel launches for basic block profiling");

// Register Device Pass
static void registerDeviceBlockProfilerPass(const PassManagerBuilder &, legacy::PassManagerBase &PM) {
  PM.add(new FluxDevicePass());
}

static RegisterStandardPasses RegisterDevicePass(PassManagerBuilder::EP_OptimizerLast, registerDeviceBlockProfilerPass);

static RegisterStandardPasses RegisterDevicePass0(PassManagerBuilder::EP_EnabledOnOptLevel0,
                                                  registerDeviceBlockProfilerPass);

// Register Host Pass as std pass
static void registerHostBlockProfilerPass(const PassManagerBuilder &, legacy::PassManagerBase &PM) {
  PM.add(new FluxHostPass());
}

// Register Host Pass early so the analysis will work better
static RegisterStandardPasses RegisterHostPass(PassManagerBuilder::EP_ModuleOptimizerEarly,
                                               registerHostBlockProfilerPass);

static RegisterStandardPasses RegisterHostPass0(PassManagerBuilder::EP_EnabledOnOptLevel0,
                                                registerHostBlockProfilerPass);
