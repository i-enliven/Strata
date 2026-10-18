// include/strata/compat/cpuid.h - the CPUID read on a host that has no cpuid.h.  GCC for Arm does not ship one
// (Clang's is empty there and its __cpuid_count reads zero, which is what the Metal build already relies on:
// src/kernels/cpu/expert_layout.cpp says at every XCR0 read that "cpuid reads zero" on arm64).  Same answer here:
// every probe in expert_layout.cpp sees a CPU with no AVX, so the engine's dispatch keeps the expert rows on the
// GPU and the i-quant models' rows on ggml-cpu's portable dots.  An x86 host takes its compiler's own header -
// CMakeLists.txt puts this directory on the include path only for a host that is not x86.
#pragma once
#define __cpuid(level, a, b, c, d) \
    ((void) (level), (a) = (b) = (c) = (d) = 0)
#define __cpuid_count(level, count, a, b, c, d) \
    ((void) (level), (void) (count), (a) = (b) = (c) = (d) = 0)
#define __get_cpuid(level, a, b, c, d) (__cpuid((level), (a), (b), (c), (d)), 0)
#define __get_cpuid_count(level, count, a, b, c, d) (__cpuid_count((level), (count), (a), (b), (c), (d)), 0)
#define __cpuid_max(level) (0)