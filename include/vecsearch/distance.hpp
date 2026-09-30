#pragma once
#include <cstddef>

namespace vs {

enum class Metric : unsigned char { L2 = 0, InnerProduct = 1 };

// Squared L2 distance and dot product. Dispatched once at startup to the
// best kernel the CPU supports (AVX2+FMA or portable scalar).
float l2_sq(const float* a, const float* b, std::size_t d) noexcept;
float dot(const float* a, const float* b, std::size_t d) noexcept;

// "Smaller is better" score for both metrics, so search code has one ordering.
inline float score(Metric m, const float* a, const float* b, std::size_t d) noexcept {
    return m == Metric::L2 ? l2_sq(a, b, d) : -dot(a, b, d);
}

// Name of the kernel selected at runtime: "avx2+fma" or "scalar".
const char* active_kernel() noexcept;

namespace detail {   // exposed for tests
float l2_sq_scalar(const float*, const float*, std::size_t) noexcept;
float dot_scalar(const float*, const float*, std::size_t) noexcept;
bool  have_avx2() noexcept;
float l2_sq_avx2(const float*, const float*, std::size_t) noexcept;
float dot_avx2(const float*, const float*, std::size_t) noexcept;
}  // namespace detail

}  // namespace vs
