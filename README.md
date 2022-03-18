# PPT-GPU: Performance Prediction Toolkit for GPUs


PPT-GPU is a scalable and flexible framework to predict the performance of GPUs running general purpose workloads. PPT-GPU can use the virtual (PTX) or the native (SASS) ISAs without sacrificing accuracy, ease of use, or portability. The tool is currently focused on NVIDIA GPUs. We plan to extend our approach to model other vendors' GPUs such as AMD and Intel.


## Papers

- For more information, check out the [SC' 21](https://doi.org/10.1145/3458817.3476221) paper ***(Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs)***.

    If you find this a helpful tool in your research, please consider citing as:

    ```
    @inproceedings{Arafa2021PPT-GPU,
      author = {Y. {Arafa} and A. {Badawy} and A. {ElWazir} and A. {Barai} and A. {Eker} and G. {Chennupati} and N. {Santhi} and S. {Eidenbenz}},
      title = {Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs},
      year = {2021},
      booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
      series = {SC '21}
    }
    ```

- The memory model is descibed in the [ICS' 20](https://doi.org/10.1145/3392717.3392761) paper.
     ```
    @inproceedings{Arafa2020PPT-GPU-MEM,
      author = {Y. {Arafa} and A. {Badawy} and G. {Chennupati} and A. {Barai} and N. {Santhi} and S. {Eidenbenz}},
      title = {Fast, Accurate, and Scalable Memory Modeling of GPGPUs Using Reuse Profiles},
      year = {2020},
      booktitle = {Proceedings of the 34th ACM International Conference on Supercomputing},
      series = {ICS '20}
    }
    ```

### See the [wiki](https://github.com/NMSU-PEARL/PPT-GPU/wiki) page for installation and usage of PPT-GPU.

***The wiki page includes the following:***

1. Infomation about the [SW & HW dependencies](https://github.com/NMSU-PEARL/PPT-GPU/wiki/SW-&-HW-Dependencies) needed for trace extraction and simualtions
2. Altenatively if you plan to use our pre-configured docker images, you can find the instructions in the [docker images & usage](https://github.com/NMSU-PEARL/PPT-GPU/wiki/Docker-Images-and-Usage) wiki page
3. [Steps](https://github.com/NMSU-PEARL/PPT-GPU/wiki/Steps-For-Running-PPT-GPU) for running the simulator 


<br />

## GPGPU Benchmarks/Applications

You can find various GPU benchmarks that can be used in your research in the following repo: [GPU Workloads](https://github.com/NMSU-PEARL/GPGPUs-Workloads)



<br />
<br />

## Classification

PPT-GPU is part of the original PPT (https://github.com/lanl/PPT) and is Unclassified and contains no Unclassified Controlled Nuclear Information. It abides with the following computer code from Los Alamos National Laboratory

  * Code Name: Performance Prediction Toolkit, C17098
  * Export Control Review Information: DOC-U.S. Department of Commerce, EAR99
  * B&R Code: YN0100000

## License

&copy 2017. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
