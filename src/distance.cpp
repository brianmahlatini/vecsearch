#include "vecsearch/distance.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define VS_X86 1
#endif

namespace vs {
namespace detail {

float l2_sq_scalar(const float* a, const float* b, std::size_t d) noexcept {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;     // 4 accumulators break the add dependency chain
    std::size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        float x0 = a[i] - b[i], x1 = a[i + 1] - b[i + 1], x2 = a[i + 2] - b[i + 2], x3 = a[i + 3] - b[i + 3];
        s0 += x0 * x0; s1 += x1 * x1; s2 += x2 * x2; s3 += x3 * x3;
    }
    for (; i < d; ++i) { float x = a[i] - b[i]; s0 += x * x; }
    return (s0 + s1) + (s2 + s3);
}

float dot_scalar(const float* a, const float* b, std::size_t d) noexcept {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    std::size_t i = 0;
    for (; i + 4 <= d; i += 4) {
        s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3];
    }
    for (; i < d; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

#ifdef VS_X86
bool have_avx2() noexcept { return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"); }

__attribute__((target("avx2,fma"))) static inline float hsum256(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}

// Two independent 8-wide FMA accumulators per iteration hide FMA latency
// (4 cycles, 2 ports on modern x86); unaligned loads are free on AVX2 cores.
__attribute__((target("avx2,fma"))) float l2_sq_avx2(const float* a, const float* b, std::size_t d) noexcept {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 8 <= d; i += 8) {
        __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    }
    float s = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < d; ++i) { float x = a[i] - b[i]; s += x * x; }
    return s;
}

__attribute__((target("avx2,fma"))) float dot_avx2(const float* a, const float* b, std::size_t d) noexcept {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= d; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= d; i += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    float s = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < d; ++i) s += a[i] * b[i];
    return s;
}
#else
bool have_avx2() noexcept { return false; }
float l2_sq_avx2(const float* a, const float* b, std::size_t d) noexcept { return l2_sq_scalar(a, b, d); }
float dot_avx2(const float* a, const float* b, std::size_t d) noexcept { return dot_scalar(a, b, d); }
#endif

}  // namespace detail

namespace {
using Fn = float (*)(const float*, const float*, std::size_t) noexcept;
struct Kernels { Fn l2; Fn dot; const char* name; };
// Resolved once, before main(); afterwards every call is one indirect call
// with a perfectly predicted target.
const Kernels kKernels = detail::have_avx2()
    ? Kernels{detail::l2_sq_avx2, detail::dot_avx2, "avx2+fma"}
    : Kernels{detail::l2_sq_scalar, detail::dot_scalar, "scalar"};
}  // namespace

float l2_sq(const float* a, const float* b, std::size_t d) noexcept { return kKernels.l2(a, b, d); }
float dot(const float* a, const float* b, std::size_t d) noexcept { return kKernels.dot(a, b, d); }
const char* active_kernel() noexcept { return kKernels.name; }

}  // namespace vs
