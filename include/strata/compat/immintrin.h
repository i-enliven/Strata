// include/strata/compat/immintrin.h - the two x86 intrinsics the host loops use, on a host without immintrin.h.
// pool.cpp spins on _mm_pause and the host raises its flags behind _mm_sfence; both files are in the CPU library on
// every host.  This is the same pair the Metal port writes (metal/include/metal_compat/immintrin.h): Arm's `yield`
// is the spin-wait hint, `dmb ishst` the store fence.  On an x86 host the real header is found first and this one is
// never read - it is not on the include path there (CMakeLists.txt: the compat directory is added only for Arm).
#pragma once
static inline void _mm_pause() { __asm__ __volatile__("yield" ::: "memory"); }
static inline void _mm_sfence() { __asm__ __volatile__("dmb ishst" ::: "memory"); }