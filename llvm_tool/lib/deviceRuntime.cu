
/// This function is responsible for determining the tracing thread(s) and
/// increasing the blocks counter There are three profiling modes:
///     a) ProfileFull: each thread is profiled, most overhead
///     b) ProfileCTA: only profile one CTA, medium overhead
///     c) ProfileWarp: all threads of one warp are profiles, least overhead
enum ProfilingMode : uint32_t { ProfileFull = 0, ProfileCTA = 1, ProfileWarp = 2 };

static int indexCountI = 1;

extern "C" __device__ void incBlockCounter(uint64_t *counters, int block_id) {
  if (threadIdx.x == blockDim.x / 2 and threadIdx.y == blockDim.y / 2 and threadIdx.z == blockDim.z / 2 and
      blockIdx.x == gridDim.x / 2 and blockIdx.y == gridDim.y / 2 and blockIdx.z == blockDim.z / 2)
    counters[block_id] += 1;
}

// Multithreaded function : (AMMAR: Added counters1 for the execution trace)
extern "C" __device__ void incBlockCounter_mt_2(uint64_t *counters, uint64_t *counters1, uint32_t block_id,
                                                uint32_t n_banks, ProfilingMode profilingMode) {

  // if (profilingMode == ProfileCTA) {
  // All Blocks but one in about the middle of the block grid will return and
  // do nothing
  // if (not(blockIdx.x == gridDim.x / 2 and blockIdx.y == gridDim.y / 2 and
  // blockIdx.z == blockDim.z / 2))
  // return;
  //} else if (profilingMode == ProfileWarp) {
  // All Blocks but one in about the middle of the block grid will return and
  // do nothing
  if (not(blockIdx.x == gridDim.x / 2 and blockIdx.y == gridDim.y / 2 and blockIdx.z == blockDim.z / 2))
    return;
  // All Warps but the one in about the middle of the thread grid will return
  // and do nothing
  uint32_t numWarps = blockDim.x * blockDim.y * blockDim.z / 32;
  uint32_t profilingWarp = numWarps / 2;
  uint32_t localThreadID = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
  uint32_t warp = localThreadID / 32;
  if (warp != profilingWarp)
    return;
  //} // else if (profilingMode == ProfileFull) {

  // get Number of Active Threads
  uint32_t lane_id;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));

  uint32_t active_threads = 0;
  uint32_t active_bitvec = __activemask();
  active_threads = __popc(active_bitvec);
  uint32_t first_active_thread = __ffs(active_bitvec) - 1;

  // first thread of warp
  if (lane_id == first_active_thread) {
    uint32_t ctaID = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    uint32_t bank = ctaID % n_banks;

    if (counters[block_id * n_banks + bank] < 0) {
      counters1[0] = 1;
    }

    atomicAdd((unsigned long long int *)&counters[block_id * n_banks + bank], (uint64_t)active_threads);

    // Execution ID: indexCountI
    // Block ID: block_id
    // counters1[0]: total number of executions for all basic blocks and for all kernels

    // atomicAdd((unsigned long long int *)&counters1[0], (uint64_t)active_threads);
    // atomicAdd((unsigned long long int *)&counters1[(block_id * n_banks) + 1 + indexCountI++],
    // (uint64_t)active_threads);

    // Saving block id at the execution id in the counters1
    if (counters1[0] == 0) {
      counters1[0] = 1;
    }
    counters1[counters1[0]] = (uint32_t)block_id + 1;
    counters1[0]++;
  }
}
