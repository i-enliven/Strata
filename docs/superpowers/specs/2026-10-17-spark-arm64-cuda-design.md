# Strata on an NVIDIA GB10 (DGX Spark): a CUDA build for arm64

Date: 2026-10-17. Branch: `feat/spark`. Machine: an NVIDIA GB10 (DGX Spark), Linux on arm64.

## Why

Strata's engine and its setup both assume an x86-64 host. On a GB10 box - one memory pool shared by the Arm CPUs
and the GPU, which is the hardware Strata's expert-cache code already names (`include/strata/core/expert_cache.hpp:76`)
- setup stops before it starts: `./setup.py --check` says "no NVIDIA GPU found" while `nvidia-smi -L` lists the card.
This is a class of machine, not one odd board: DGX Spark, and anything else whose `nvidia-smi` reports no separate
`memory.total`.

The Metal port already made the decision this port needs: **the GPU does the model's math, and the CPU kernel
library is reduced to what the engine cannot do without.** The Spark build follows that template, keeping the CUDA
side whole.

## The machine, measured 2026-10-17

| | |
| --- | --- |
| GPU | NVIDIA GB10, compute capability 12.1 (`sm_121`), driver 595.99.02 |
| `nvidia-smi` memory | `[N/A]` - there is no separate VRAM to report |
| System RAM | 121 GB total; 41 GB available while a TabbyAPI (exl3) server holds the rest. The whole pool is Strata's when that server is stopped. |
| CPUs | 20, arm64 (aarch64) |
| Toolkit | CUDA 13.0.88 at `/usr/local/cuda-13.0`; `nvcc --list-gpu-code` includes `sm_121` |
| Host compilers | gcc 13.3.0, cmake 3.28.3; no `ninja` on PATH (setup installs cmake and ninja into `.venv`) |
| Disk | 660 GB free |
| Engine source | 67 `.cu` files, 47,715 lines of CUDA under `src/`, 161 headers under `src/` and `include/` |

## What "works" means

**Goals**

1. `./setup.sh --check` sees the GB10 and reports it correctly (name, compute capability, the memory it can use).
2. `./setup.sh` compiles the engine here - there is no ready-made engine for arm64 - and starts a model.
3. A chat request through the server returns an answer, and the run is measured: decode and prompt tokens/s,
   recorded the way `docs/COMMUNITY_BENCHMARKS.md` asks, against a same-day run of this branch.
4. Nothing about an x86 install changes: same files, same flags, same engine.

**Non-goals**

- CPU-computed experts on arm64. They exist to keep experts out of a 12 GB card's VRAM; a GB10 has no separate VRAM
  to free. The AVX-512 / AVX-2 / AVX1 expert kernels stay out of the arm64 build.
- The CPU prefill share (`STRATA_PREFILL_CPU_SHARE`, on by default for NVIDIA since 0.1.41) on arm64. Same reason:
  it is a way of using spare CPU compute next to a small card.
- NEON replacements for the AVX kernels. If a later measurement wants them, that is a separate change.
- A ready-made arm64 engine in the releases, and any change to what the x86 release ships.

**Anti-goals**

- An arm64 build that links but stops at run time on a stub, with a message that names an intrinsics file instead of
  the entry point.
- A port that quietly changes the x86 path - a shared header edited for arm64's benefit, a default flipped for
  everyone.
- A doc claim without a run behind it.

## Constraints

From `AGENTS.md`: the new path is opt-in in the sense that only an arm64 host takes it; the default path stays
byte-identical; every backend a touched file builds (CUDA, HIP, SYCL, and Metal for anything under `metal/`) is
built before review; docs keep plain words and measured numbers.

`CMakeLists.txt` is one file for all backends, so its changes must be inert on x86 by construction, not by testing.

## The template: what the Metal port already does

| Metal | Where | Spark equivalent |
| --- | --- | --- |
| Keeps three files from the CPU kernel directory: `pool.cpp`, `expert_layout.cpp`, `native_expert.cpp` | `metal/CMakeLists.txt:100` | Same three, plus one new file for the entry points the AVX files would have provided |
| Compiles them on arm64 with a five-line `immintrin.h`: `_mm_pause` to `yield`, `_mm_sfence` to `dmb ishst` | `metal/include/metal_compat/immintrin.h` | A copy under `include/strata/compat/`, on the include path only for the arm64 branch |
| Stubs 30 CPU kernel entry points so they stop with the entry point's name | `metal/src/engine/stubs.cpp:155-180`, `metal/src/engine/stubs_144.cpp:40-55` | The same list, in one new source file |
| Links `ggml-cpu` and `ggml-base` for the native-expert path | `metal/CMakeLists.txt:113` | Same; ggml builds on arm64 with its own NEON dispatch |

Twelve files under `src/kernels/cpu/` include `immintrin.h`. Of the four engine host files that also include it -
`session.cpp`, `verify.cpp`, `graph.cpp`, `mtp.cpp` - two intrinsics are used between them: `_mm_pause` (10 uses) and
`_mm_sfence` (13). The fp16 conversion intrinsics are only in `src/prefill/mmq_nvfp4_parity.cpp`, which is a test
target (`CMakeLists.txt:1729`) and is not built with `-DSTRATA_BUILD_TESTS=OFF`. So the shim's two functions are the
whole x86 surface of the engine's host side.

Two runtime paths matter here, and both already have a non-AVX answer:

- **The router's CPU dot**, which runs on every token, falls through to a plain-C++ loop written out in
  `src/core/expert_source.cpp:2395-2406` when `cpu_avx2_ok()` and `cpu_avx1_ok()` are both false. It is a prefetch
  estimate only - "the output does not depend on it" (that file's comment at 2386-2388).
- **The expert row dispatch** `q2_rows_any` (`src/kernels/cpu/expert_layout.cpp:356-361`) has three rungs and no
  fourth: with no AVX it calls the AVX1 function. On arm64 that function must exist and must stop with a clear
  message rather than being an undefined symbol, and nothing in the default arm64 run may reach it.

## The design

**A. setup.py sees this PC.** Two fixes, both small and both wrong-on-x86-safe:

- `gpus()` (`setup.py:999-1008`) drops a card whose `memory.total` is not a number: `float("[N/A]")` raises and
  `except ValueError: continue` swallows the row. For a device with no reported memory, take the size from the
  system the way the engine does for the expert cache - `MemAvailable` on a unified-memory system - and mark the
  card so the rest of setup knows its "VRAM" is shared RAM.
- `cpu_info()` (`setup.py:883`) reads `/proc/cpuinfo`'s `flags:` line and crashes when there is none; arm64 spells
  the line `Features`. Return `(name, False, False)` when the line is absent, taking the name from `Model` /
  `Hardware`. `docker-entrypoint.sh:12` already guards the same file this way.

With `avx512` false, the Q2_0 AVX-512 repack (`setup.py:6795-6799`, "~40 GB written") is skipped by its own
condition - no change needed there.

**B. setup.py builds here instead of downloading.** There is no arm64 asset in the releases, so the prebuilt probe
(`get_prebuilt`, `setup.py:3620-3632`) must not be tried on arm64: go straight to the local compile, which
`build_engine` (`setup.py:4046+`) already knows how to run. The x86 download path is untouched.

**C. CMake: an arm64 branch in `strata_kernels_cpu`.** Where `CMakeLists.txt:1361-1373` adds the AVX files and their
`-mavx512*` / `-mavx2` flags, split on the host processor: x86 keeps exactly what it does today; arm64 builds the
three files Metal builds plus the new entry-point file, with no `-m` flags and `include/strata/compat` on the include
path. `strata_engine` links this library `PUBLIC` (`CMakeLists.txt:558`), so it must exist as a target on both paths.
The same split covers the ggml block at 1465-1470, which adds six more AVX files.

**D. The entry-point file.** One new source, `src/kernels/cpu/arm64_unavailable.cpp`, holding the 30 names Metal
stubs. The `*_supported()` predicates (`iq128_supported`, `iq128_down_supported`) return false, so the runtime
dispatcher never selects a CPU kernel; the row kernels stop with the entry point's name and a line saying this build
has no CPU expert kernels, in the words `cpu_require_expert_support()` already uses (`expert_layout.cpp:349-352`).

**E. Runtime defaults on arm64.** The CPU prefill share is armed by default on NVIDIA
(`src/program/generate.cpp:7898`, `14900`), and `Prefill::arm_cpu_share` already compiles that default out for HIP
(`src/prefill/prefill.cpp:2229`: `#if !defined(STRATA_USE_HIP)`). arm64 joins that guard, so the share stays disarmed
unless `STRATA_PREFILL_CPU_SHARE` asks for it; asking for it says plainly that this build has no CPU expert kernels
rather than reaching a stub. The alternative - setup writing the variable into the run config's `env` block
(`setup.py:4523`, `4879`) - leaves the engine untouched but only covers starts that go through the config, so the
compile-time guard is the one to take.

**F. Docs and tests.** `docs/SPARK.md` with what was measured and what was not, linked from `docs/INSTALL.md` and
`README.md`'s platform list; `docs/TEST_REQUESTS.md` gains the hardware this was and was not tested on.
`tools/test_setup_spark.py`, which runs with no GPU and no download like the other setup tests: the `[N/A]` memory
row, a `flags`-less `/proc/cpuinfo`, no prebuilt probe on arm64, the repack condition, and a source check that the
CMake arm64 branch cannot be reached from the x86 branch.

## How it gets checked

1. `python3 tools/test_setup_spark.py` and the existing setup tests, before and after each change. Here, three setup
   test modules already fail on this machine for the two reasons above - `test_setup_choices` (5 errors),
   `test_setup_amd` (1 error) - and A should clear those; `test_setup_engine_hash` (2 failures) reaches the network
   through `setup.py:3624` and is a separate matter.
2. A real configure and build on this box for `sm_121`, and the same configure on x86 assumptions checked by reading
   the generated compile commands: the x86 flag list must come out unchanged.
3. HIP and SYCL configured on this branch (AGENTS.md asks for every backend a touched file builds). On an NVIDIA-only
   box that means configure, not run.
4. A run: Q2_0 first, `--check`, then a chat request, then `tools/strata_bench.py` for the numbers, with the memory
   pool to itself.

## Risks

- **The link surface may be wider than Metal's 30.** Metal's list came from its own link failure
  (`metal/tools/gen_stubs.py` reads a failed link log). The CUDA build compiles different host code, so the arm64
  build may need more names. The plan should expect one link round-trip, and generate the list the same way rather
  than guessing it.
- **The pip CUDA wheels may not exist for arm64** (`CUDA_WHEELS` at `setup.py:183`: `nvidia-cublas==13.0.2.14`,
  `nvidia-cuda-runtime==13.0.96`). This box has the toolkit at `/usr/local/cuda-13.0`, so the system libraries can be
  used instead; which of the two setup does, and whether `cuda_lib_dirs()` (`setup.py:2078`) needs an arm64 case, is
  an implementation question with a one-command answer.
- **The router's plain-C++ dot on 20 Arm cores** may cost real time per token. It is a prefetch estimate, so it can
  be capped or skipped; measure before optimising.
- **121 GB is not 121 GB while something else runs.** Every number recorded must say what else held memory.

## Calls I made (not the partner's)

- Target Q2_0 first: it is setup's default and the README's fastest row, and its AVX-512 repack is skipped by its own
  condition once `cpu_info()` is fixed.
- Gate the CMake branch on the host processor rather than a new option, so an arm64 user types nothing new and an
  x86 user cannot reach the new code.
- Copy the five-line shim into `include/strata/compat/` rather than share `metal/include/metal_compat/`, so the Metal
  build stays byte-identical; unifying the two is a later, separate change.
- Key each new branch on the fact it is about, not on "is this a Spark": the CPU kernel files on the host processor
  (a compile-time fact), the memory figure on the device reporting no memory and being integrated (a device fact, the
  same test `expert_cache.cpp:42` makes). A Graviton host with a discrete card keeps the CPU kernels and its own VRAM
  figures; a Spark gets neither.
- Stub rather than port the CPU expert kernels, following Metal.
- `docs/SPARK.md` as a new file, so no existing doc's measured tables are edited with numbers from hardware the
  maintainers do not have.