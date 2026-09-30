#pragma once
// Minimal static-partition parallel_for over [0, n). Work items in vector
// search are uniform (one query each), so static chunks beat a work queue:
// no shared counter, no contention, perfect locality per thread.
#include <algorithm>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace vs {

inline unsigned default_threads() noexcept {
    unsigned t = std::thread::hardware_concurrency();
    return t == 0 ? 1u : t;
}

// fn(begin, end) is invoked on disjoint ranges. The first exception thrown by
// any worker is rethrown on the calling thread after all workers join.
template <class Fn>
void parallel_for(std::size_t n, unsigned threads, Fn&& fn) {
    if (n == 0) return;
    if (threads == 0) threads = default_threads();
    const std::size_t t = std::min<std::size_t>(threads, n);
    if (t == 1) { fn(std::size_t{0}, n); return; }

    std::exception_ptr err;
    std::mutex err_mu;
    {
        std::vector<std::jthread> pool;
        pool.reserve(t - 1);
        const std::size_t chunk = n / t, rem = n % t;
        std::size_t begin = 0;
        auto run = [&](std::size_t b, std::size_t e) {
            try { fn(b, e); } catch (...) {
                std::lock_guard lk(err_mu);
                if (!err) err = std::current_exception();
            }
        };
        std::size_t first_end = 0;
        for (std::size_t i = 0; i < t; ++i) {
            const std::size_t end = begin + chunk + (i < rem ? 1 : 0);
            if (i == 0) first_end = end;
            else pool.emplace_back(run, begin, end);
            begin = end;
        }
        run(0, first_end);        // calling thread does its share too
    }                             // jthreads join here
    if (err) std::rethrow_exception(err);
}

}  // namespace vs
