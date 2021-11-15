#include </software/GPU/Cuda/10.1/include/cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <unordered_map>
#include <vector>

#include <bits/stdc++.h>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

// TODO implement logging/log levels

#define CUDA_CHECK(ERRORCODE)                                                                                          \
  {                                                                                                                    \
    cudaError_t error = ERRORCODE;                                                                                     \
    if (error != 0) {                                                                                                  \
      cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__    \
           << "\n";                                                                                                    \
    }                                                                                                                  \
  }

// Allocating the array to be the number of basic blocks * array_size
#define array_size 100000

extern "C" uint64_t *__attribute__((weak)) createBBCounterMemory(size_t len) {

  uint64_t *d_ptr;

  CUDA_CHECK(cudaMalloc(&d_ptr, len * sizeof(uint64_t)));

  if (d_ptr == nullptr) {
    cerr << "Error allocating memory for basic block counters. Exiting...\n";
    exit(-1);
  }

  CUDA_CHECK(cudaMemset(d_ptr, 0, len * sizeof(uint64_t)));

  return d_ptr;
}

// Creating and Allocating our new array
extern "C" uint64_t *__attribute__((weak)) createBBCounterMemory1(size_t len) {

  uint64_t *t_ptr;

  CUDA_CHECK(cudaMalloc(&t_ptr, len * array_size * sizeof(uint64_t)));

  if (t_ptr == nullptr) {
    cerr << "Error allocating memory for basic block counters. Exiting...\n";
    exit(-1);
  }

  CUDA_CHECK(cudaMemset(t_ptr, 0, len * array_size * sizeof(uint64_t)));
  return t_ptr;
}

__attribute__((weak)) bool g_profilingModePolled = false;
__attribute__((weak)) uint32_t g_profilingMode = 0;

extern "C" uint32_t __attribute__((weak)) getProfilingMode() {

  if (g_profilingModePolled)
    return g_profilingMode;

  char *profilingMode;
  profilingMode = getenv("MEKONG_PROFILINGMODE");
  if (profilingMode != NULL) {
    g_profilingMode = atoi(profilingMode);
    g_profilingModePolled = true;
  }

  return g_profilingMode;
}

extern "C" struct timeval *__attribute__((weak)) startClock() {
  struct timeval *t = new struct timeval();
  gettimeofday(t, nullptr);
  return t;
}

// Identifier for multi-executed kernels
static int kernelCount = 1;

// TODO print when debug env is set
extern "C" void __attribute__((weak))
serializeBBCounter(uint64_t *d_blockList, uint64_t *d_blockList1, size_t len, uint32_t n_banks, const char *kernelname,
                   uint64_t gridXY, uint32_t gridZ, uint64_t blockXY, uint32_t blockZ, uint64_t shmsize,
                   cudaStream_t stream, struct timeval *t1) {
  CUDA_CHECK(cudaStreamSynchronize(stream));

  struct timeval t2;
  gettimeofday(&t2, nullptr);

  unsigned long diff = (t2.tv_sec - (t1->tv_sec)) * 10e6 + (t2.tv_usec - (t1->tv_usec));
  free(t1);

  uint64_t *h_blockList = (uint64_t *)malloc(len * sizeof(uint64_t));
  uint64_t *h_blockList1 = (uint64_t *)malloc(len * array_size * sizeof(uint64_t));

  // Copy basic block counter to host memory
  CUDA_CHECK(cudaMemcpy(h_blockList, d_blockList, len * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_blockList1, d_blockList1, len * array_size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

  for (int block_ID = 0; block_ID < len / n_banks; ++block_ID) {
    uint64_t sum = 0;
    uint64_t sum1 = 0;
    for (int bank = 0; bank < n_banks; ++bank) {
      if (__builtin_add_overflow(sum, h_blockList[block_ID * n_banks + bank], &sum)) {
        // todo
        cerr << "CUDA FLUX: OVERFLOW!\n";
      }
      // Getting if our new counter overflows
      if (__builtin_add_overflow(sum1, h_blockList1[block_ID * n_banks + bank], &sum1)) {
        // todo
        cerr << "CUDA FLUX: OVERFLOW1!\n";
      }
    }

    // Store result to column for bank zero
    h_blockList[block_ID * n_banks + 0] = sum;
    h_blockList1[block_ID * n_banks + 0] = sum1;
  }

  ofstream output("bbc.txt", ios::app);

  // TODO dump version
  if (output.tellp() == 0) {
    output << "# CUDA_Flux_Version=0.3\n";
  }

  output << string(kernelname);

  // Reading number of executions for every block
  for (int block_ID = 0; block_ID < len / n_banks; ++block_ID) {
    // Read result from column for bank zero
    output << ", " << h_blockList[block_ID * n_banks + 0] / 32.0;
  }

  // Creating trace file for number of executions and trace file per kernel execution

  mkdir("ptx_traces", 0777);
  ofstream output_kernel("ptx_traces/kernel_" + std::to_string(kernelCount) + ".ptx", ios::app);

  kernelCount++;

  // output1 << "\n\n New Counter\n";

  // output1 << "\nTotal Number of Executions: " << h_blockList1[0] << "\n\n";

  // Reading out file for every kernel and saving it to a vector
  std::string line;
  int ind = 0;
  std::unordered_map<int, std::vector<std::string>> bb_inst_map;
  ifstream outfile(string(kernelname) + ".out");
  if (outfile.is_open()) {
    while (getline(outfile, line)) {
      if (std::strcmp(line.c_str(), "--") == 0) {
        ind++;
      } else {
        bb_inst_map[ind].push_back(line);
      }
    }
    outfile.close();
  } else {
    cout << "Unable to load file for Kernel: " << string(kernelname) << "\n";
  }

  // Reading instructions' lines per basic block and converting it to a vector
  std::vector<std::string> trace_lines;
  for (ind = 1; ind < h_blockList1[0]; ind++) {
    if (h_blockList1[ind] != 0) {
      for (auto &bbLine : bb_inst_map[h_blockList1[ind]]) {
        trace_lines.push_back(bbLine);
      }
    }
  }

  // Reading Dependencies and comparing them for three main cases
  std::unordered_map<std::string, int> dependencies;
  std::string temp = "";
  for (ind = 0; ind < trace_lines.size(); ind++) {
    temp = "";
    char *pch;
    std::vector<std::string> tempvec;
    char *tli = strdup(trace_lines.at(ind).c_str());
    pch = std::strtok(tli, " ");
    while (pch != NULL) {
      tempvec.push_back(std::string(pch));
      pch = std::strtok(NULL, " ");
    }
    temp = temp + tempvec.at(0) + " ";
    if (tempvec.size() == 2) {
      if (tempvec.at(1).find("[") != std::string::npos) {
        // Reading Registers saving memory addresses
        std::size_t index = 0;
        index = tempvec.at(1).find("..");
        if (index != std::string::npos)
          tempvec.at(1).erase(0, index + 2);
        index = 0;
        index = tempvec.at(1).find("..");
        if (index != std::string::npos)
          tempvec.at(1).erase(index, tempvec.at(1).length() - 1);
      }
      if (dependencies.find(tempvec.at(1)) != dependencies.end()) {
        temp = temp + std::to_string(dependencies[tempvec.at(1)]) + " ";
      }
    } else if (tempvec.size() > 2) {
      for (int j = 2; j < tempvec.size(); j++) {
        if (tempvec.at(j).find("[") != std::string::npos) {
          std::size_t index = 0;
          index = tempvec.at(j).find("..");
          if (index != std::string::npos)
            tempvec.at(j).erase(0, index + 2);
          index = 0;
          index = tempvec.at(j).find("..");
          if (index != std::string::npos)
            tempvec.at(j).erase(index, tempvec.at(j).length() - 1);
          // temp = temp + " DEBUG " + tempvec.at(j) + " DEBUG ";
        }
        if (dependencies.find(tempvec.at(j)) != dependencies.end()) {
          temp = temp + std::to_string(dependencies[tempvec.at(j)]) + " ";
        }
      }
      if (tempvec.at(1).find("|") != std::string::npos) {
        char *dch;
        char *tv1 = strdup(tempvec.at(1).c_str());
        dch = std::strtok(tv1, "|");
        while (dch != NULL) {
          if (std::string(dch).find("[") != std::string::npos) {
            if (std::string(dch).find("%") != std::string::npos) {
              std::size_t index = 0;
              index = std::string(dch).find("..", index);
              if (index != std::string::npos)
                std::string(dch).erase(0, index + 2);
              index = 0;
              index = std::string(dch).find("..", index);
              if (index != std::string::npos)
                std::string(dch).erase(index, std::string(dch).length() - 1);
              if (dependencies.find(std::string(dch)) != dependencies.end()) {
                temp = temp + std::to_string(dependencies[std::string(dch)]) + " ";
              }
            }
          } else {
            dependencies[std::string(dch)] = ind;
          }
          dch = std::strtok(NULL, "|");
        }
      } else {
        if (tempvec.at(1).find("[") != std::string::npos) {
          std::size_t index = 0;
          index = tempvec.at(1).find("..");
          if (index != std::string::npos)
            tempvec.at(1).erase(0, index + 2);
          index = 0;
          index = tempvec.at(1).find("..");
          if (index != std::string::npos)
            tempvec.at(1).erase(index, tempvec.at(1).length() - 1);
          if (dependencies.find(tempvec.at(1)) != dependencies.end()) {
            temp = temp + std::to_string(dependencies[tempvec.at(1)]) + " ";
          }
        } else {
          dependencies[tempvec.at(1)] = ind;
        }
      }
    }
    if (!temp.empty()) {
      output_kernel << temp << "\n";
    }
  }

  output << "\n";
  output.flush();
  output_kernel.flush();

  CUDA_CHECK(cudaFree(d_blockList));
  CUDA_CHECK(cudaFree(d_blockList1));
  free(h_blockList);
  free(h_blockList1);
}

extern "C" void __attribute__((weak)) mekong_WritePTXBlockAnalysis(bool *isWritten, const char *analysis) {

  if (*isWritten)
    return;

  ofstream output("PTX_Analysis.yml", ios::app);

  output << analysis;

  *isWritten = true;
}
