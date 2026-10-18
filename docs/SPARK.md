# NVIDIA DGX Spark (GB10, sm_121, arm64): build from source on Linux

Strata runs on the NVIDIA GB10 in the DGX Spark (an Arm aarch64 SoC: 20 CPU cores, one pool of LPDDR5X memory shared by the CPU and the GPU, `sm_121`) through the CUDA backend. **Status: experimental.** Everything here was built and measured on one machine (NVIDIA DGX Spark, 121.6 GiB RAM, Ubuntu 24.04, Linux kernel 7.0.0-1019-nvidia, driver 595.99.02, CUDA 13.0.88 / nvcc 13.0).

## 1. What setup and the engine do differently here

- **The engine is compiled locally.** The release downloads carry `strata-linux-x64.zip` and `strata-windows-x64.zip` for x86-64 only; `get_prebuilt()` detects an arm64 host, skips remote probing, and announces:
  `No ready-made engine for arm64: it is compiled here (docs/SPARK.md)`.
  The installer builds target `strata` for `sm_121` using CMake and Ninja.
- **Unified memory.** `nvidia-smi` on an integrated GB10 reports `[N/A]` for total memory. Setup detects `mem_na`, queries the CUDA runtime via `cudaDevAttrIntegrated` (attribute 18), and treats the GPU memory as the available system RAM less 6 GB kept for the OS (`UMA_OS_LEFT_GB`):
  `GPU: NVIDIA GB10, unified memory: 116 GB shared with the RAM, 116 GB usable by the GPU, compute capability 12.1, driver 595.99.02`.
- **No PCIe probe.** A GB10 resides directly on the SoC; PCIe link rate and width probes are skipped via `pcie_applies()` to avoid phantom x1 reports.
- **Arm-native CPU kernel library.** `STRATA_HOST_X86` in `CMakeLists.txt` isolates the x86 AVX/SSE kernel sources (`expert.cpp`, `q2_avx2.cpp`, `iq_*.cpp`, etc.). On arm64, `strata_kernels_cpu` is built using `pool.cpp`, `expert_layout.cpp`, `native_expert.cpp` (using ggml-cpu portable row dots), and `arm64_unavailable.cpp` with compatibility headers for `_mm_pause`, `_mm_sfence` (`include/strata/compat/immintrin.h`) and `cpuid.h` (`include/strata/compat/cpuid.h`).
- **No CPU prefill share.** `Prefill::arm_cpu_share` does not arm `g_stream_min_share` on `__aarch64__`: there are no CPU AVX kernels to execute CPU-side expert rows. If explicitly requested via `STRATA_PREFILL_CPU_SHARE=1`, the engine notes that no CPU expert kernels are present and routes all experts to the GPU.
- **CPU identification.** The CPU is displayed as `aarch64 (Arm)`, and AVX2/SSE4.2 warnings/floors are skipped. In `src/program/generate.cpp`, the x86 AVX2 ISA check is bypassed on `__aarch64__`, matching the Apple Silicon Metal logic.

## 2. Requirements & Setup

On Ubuntu 24.04 (PEP 668 managed environment), run the setup script rather than direct `python3 setup.py`:

```bash
./setup.sh --yes --family qwen --model IQ3_S --no-start
```

Ensure the CUDA toolkit (`nvcc`) is available in `/usr/local/cuda` and `/usr/local/cuda/bin` is in `PATH`. Setup will compile the engine locally and configure `run-iq3_s.sh`.

To start the server:
```bash
./run-iq3_s.sh
```

## 3. Measured performance

Measured on 2026-10-18 using `tools/strata_bench.py --long` against a locally running server on NVIDIA DGX Spark (GB10, 121.6 GiB RAM), model `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, 131,072 context, int8 KV cache, MTP draft layer enabled, all 24,576 experts resident in unified memory. Median and range across 3 runs:

| Test | Metrics |
|---|---|
| Chat decode | **54.4 tok/s** (47.8 - 72.3 tok/s), TTFT **0.43 s** (0.39 - 0.47 s) |
| 4K prompt | **1472.9 tok/s** (1373.2 - 1525.0 tok/s), TTFT **2.76 s** |
| 16K prompt | **1674.1 tok/s** (1642.2 - 1678.4 tok/s), TTFT **9.63 s** |
| 32K prompt | **1700.5 tok/s** (1698.9 - 1701.1 tok/s), TTFT **18.91 s** |
| Agent session | **59.5 tok/s** decode (41.0 - 70.8 tok/s), **68.8%** cache hit rate |
| Memory usage | ~105 GB utilized by `strata` during active inference |

Full report and raw JSON: [bench/results/2026-10-18-community-gb10-qwen3-8-flash-next-iq3-s/](bench/results/2026-10-18-community-gb10-qwen3-8-flash-next-iq3-s/).

## 4. Not tested

- HIP and SYCL on arm64: ROCm and ICX are not installed on this host.
- Windows on arm64 (WOA).
- Multi-GPU configurations mixing a GB10 SoC with discrete PCIe cards.
- Vision/multimodal encoder (`images: off`).
- Other quantizations (Q2_0, IQ2_XS, IQ3_XXS, NVFP4).
- Standalone AVX parity tests (`iq_avx2_parity`, etc.) on arm64.
