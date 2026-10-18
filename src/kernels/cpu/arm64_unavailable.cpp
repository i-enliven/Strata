// src/kernels/cpu/arm64_unavailable.cpp - the CPU expert entry points the AVX kernel files export, on a host that
// has no AVX.  Strata's CPU experts (src/kernels/cpu/expert.cpp, q2_avx2.cpp, q2_avx1.cpp, iq_*.cpp, kq_*.cpp,
// nvfp4_*.cpp) are written with AVX-512 / AVX2 / SSE intrinsics; an Arm host - the GB10 in a DGX Spark - cannot
// compile them, so CMakeLists.txt leaves them out of strata_kernels_cpu there and the host code still calls these.
// Every call site is behind one of the probes in expert_layout.cpp (cpu_avx512_ok(), cpu_avx2_ok(), cpu_avx1_ok()),
// which on this host read no AVX and answer false, so none of the row kernels below should ever be reached: they
// name themselves and stop rather than return numbers nobody computed.  The four queries are different - the pool
// asks them before choosing a path - and they answer "no", which sends the work to the GPU or to ggml-cpu's
// portable dots (src/kernels/cpu/native_expert.cpp, built here).  The Metal port does the same thing for the
// kernels it has no shader for (metal/src/engine/stubs.cpp, metal/src/engine/missing.cpp).
//
// The list is the linker's: every symbol below is one it reported undefined when the engine was first linked on a
// GB10.  Add a stub the same way after excluding another file; delete one after porting it.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"

#include <cstdio>
#include <cstdlib>

namespace strata::kernels::cpu {
namespace {

[[noreturn]] void no_avx(const char* who) {
    std::fprintf(stderr, "strata: %s is not in this build: it is an AVX kernel and this CPU has none, so the engine "
                         "should not have asked for it (the expert rows run on the GPU, or through ggml-cpu on this "
                         "host - docs/SPARK.md)\n", who);
    std::abort();
}

}  // namespace

// ---- the Q2_0 / S2 experts (src/kernels/cpu/expert.cpp, q2_avx2.cpp, q2_avx1.cpp) ----
void act_quant_q8_1(const float*, int, ActQ&) { no_avx("act_quant_q8_1"); }
void act_quant_q8_1_avx2(const float*, int, ActQ&) { no_avx("act_quant_q8_1_avx2"); }
void act_quant_q8_1_avx1(const float*, int, ActQ&) { no_avx("act_quant_q8_1_avx1"); }
void s2_expert_vnni_q(const uint8_t*, const ActQ&, float*, ExpertScratch&) { no_avx("s2_expert_vnni_q"); }
void s2_expert_gu_rows(const uint8_t*, const ActQ&, float*, int, int) { no_avx("s2_expert_gu_rows"); }
void s2_expert_down_rows(const uint8_t*, const ActQ&, float*, int, int) { no_avx("s2_expert_down_rows"); }
void s2_expert_gu_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    no_avx("s2_expert_gu_rows_multi");
}
void s2_expert_down_rows_multi(const uint8_t*, const ActQ* const*, int, float* const*, int, int) {
    no_avx("s2_expert_down_rows_multi");
}
void q2_0_gguf_rows_multi(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    no_avx("q2_0_gguf_rows_multi");
}
void q2_0_gguf_rows_multi_avx2(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    no_avx("q2_0_gguf_rows_multi_avx2");
}
void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    no_avx("q2_0_gguf_rows_multi_avx2_legacy");
}
void q2_0_gguf_rows_multi_avx1(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    no_avx("q2_0_gguf_rows_multi_avx1");
}
void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t*, size_t, int, const ActQ* const*, int, float* const*, int, int) {
    no_avx("q2_0_gguf_rows_multi_avx1_legacy");
}
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) { no_avx("bf16_rows_dot_multi"); }
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    no_avx("bf16_rows_dot_multi_avx1");
}
void q8k_quant_avx2(const float*, void*, int64_t) { no_avx("q8k_quant_avx2"); }
void q8k_quant_avx1(const float*, void*, int64_t) { no_avx("q8k_quant_avx1"); }

// ---- the i-quant and K-quant rows (src/kernels/cpu/iq_*.cpp, kq_avx2.cpp) ----
// The four queries answer "no": native_expert.cpp then takes ggml-cpu's portable dots instead of these.
bool iq128_supported(int) noexcept { return false; }
bool iq128_down_supported(int) noexcept { return false; }
void iq128_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    no_avx("iq128_gu_rows");
}
void iq128_down_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    no_avx("iq128_down_rows");
}
void iq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    no_avx("iq256_gu_rows");
}
void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    no_avx("iq256_gu_rows_exact_one");
}
void iq512_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    no_avx("iq512_gu_rows");
}
void iq4nl256_down_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    no_avx("iq4nl256_down_rows");
}
void kq256_gu_rows(int, const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int) {
    no_avx("kq256_gu_rows");
}
void kq256_rows(int, const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int) {
    no_avx("kq256_rows");
}

// ---- the NVFP4 experts (src/kernels/cpu/nvfp4_avx512.cpp, nvfp4_avx2.cpp) ----
bool nvfp4_512_fits(int) { return false; }
bool nvfp4_256_fits(int) { return false; }
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int,
                       float, float) { no_avx("nvfp4_512_gu_rows"); }
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    no_avx("nvfp4_512_rows");
}
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int,
                       float, float) { no_avx("nvfp4_256_gu_rows"); }
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    no_avx("nvfp4_256_rows");
}

}  // namespace strata::kernels::cpu