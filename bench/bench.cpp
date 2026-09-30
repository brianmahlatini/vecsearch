// vecsearch benchmark: kernel throughput, flat (exact) QPS, IVF QPS/recall.
// usage: vs_bench [n=100000] [dim=128] [nq=1000] [threads=0(auto)]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/ivf_index.hpp"
#include "vecsearch/parallel.hpp"

using namespace vs;
using Clock = std::chrono::steady_clock;

static std::vector<float> clustered(std::size_t n, std::size_t d, std::size_t clusters, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::mt19937_64 crng(1234);   // shared centers for db and queries
    std::vector<float> centers(clusters * d);
    for (auto& x : centers) x = g(crng) * 4.f;
    std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
    std::vector<float> v(n * d);
    for (std::size_t i = 0; i < n; ++i) {
        const float* c = centers.data() + pick(rng) * d;
        for (std::size_t j = 0; j < d; ++j) v[i * d + j] = c[j] + g(rng);
    }
    return v;
}

static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 100000;
    const std::size_t d = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 128;
    const std::size_t nq = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 1000;
    const unsigned th = argc > 4 ? unsigned(std::strtoul(argv[4], nullptr, 10)) : 0;
    const std::size_t k = 10;
    std::printf("n=%zu dim=%zu nq=%zu k=%zu threads=%u kernel=%s\n", n, d, nq, k,
                th ? th : default_threads(), active_kernel());

    auto db = clustered(n, d, 256, 1), qs = clustered(nq, d, 256, 2);

    // 1. Raw kernel throughput (vectors in L1: measures compute, not memory).
    {
        const std::size_t reps = 2'000'000;
        volatile float sink = 0;
        for (int which = 0; which < 2; ++which) {
            auto fn = which == 0 ? detail::l2_sq_scalar : (detail::have_avx2() ? detail::l2_sq_avx2 : detail::l2_sq_scalar);
            auto t0 = Clock::now();
            float acc = 0;
            for (std::size_t r = 0; r < reps; ++r) acc += fn(db.data() + (r & 63) * d, qs.data(), d);
            auto t1 = Clock::now();
            sink = acc;
            const double gflops = double(reps) * double(d) * 3 / secs(t0, t1) / 1e9;   // sub + mul + add
            std::printf("l2 kernel %-8s : %6.1f M dist/s  %6.1f GFLOP/s\n", which == 0 ? "scalar" : "avx2",
                        double(reps) / secs(t0, t1) / 1e6, gflops);
        }
        (void)sink;
    }

    // 2. Flat exact search.
    FlatIndex flat(d);
    flat.add(db.data(), n);
    auto t0 = Clock::now();
    auto truth = flat.search_batch(qs.data(), nq, k, th);
    auto t1 = Clock::now();
    const double flat_qps = double(nq) / secs(t0, t1);
    std::printf("flat      : %8.0f QPS  (%.2f ms/query, %.2f G dist/s)\n", flat_qps, 1e3 / flat_qps,
                flat_qps * double(n) / 1e9);

    // 3. IVF.
    IvfParams p;
    p.nlist = 1024;
    p.train_iters = 10;
    IvfIndex ivf(d, p);
    t0 = Clock::now();
    ivf.train(db.data(), n, th);
    ivf.add(db.data(), n, th);
    t1 = Clock::now();
    std::printf("ivf build : %.2f s (nlist=%zu)\n", secs(t0, t1), p.nlist);
    for (std::size_t nprobe : {1u, 4u, 8u, 16u, 32u, 64u}) {
        t0 = Clock::now();
        auto r = ivf.search_batch(qs.data(), nq, k, nprobe, th);
        t1 = Clock::now();
        double rec = 0;
        for (std::size_t q = 0; q < nq; ++q) {
            std::set<std::int64_t> t;
            for (auto& h : truth.row(q)) t.insert(h.id);
            for (auto& h : r.row(q)) rec += double(t.count(h.id));
        }
        rec /= double(nq * k);
        const double qps = double(nq) / secs(t0, t1);
        std::printf("ivf np=%-3zu: %8.0f QPS  recall@10=%.3f  speedup vs flat %.1fx\n", nprobe, qps, rec,
                    qps / flat_qps);
    }
}
