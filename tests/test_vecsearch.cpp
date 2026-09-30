#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/ivf_index.hpp"
#include "vecsearch/topk.hpp"

using namespace vs;

namespace {

std::vector<float> random_vectors(std::size_t n, std::size_t d, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> v(n * d);
    for (auto& x : v) x = g(rng);
    return v;
}

// Gaussian mixture: realistic embedding-like data with cluster structure.
std::vector<float> clustered(std::size_t n, std::size_t d, std::size_t clusters, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> centers(clusters * d);
    for (auto& x : centers) x = g(rng) * 4.f;
    std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);
    std::vector<float> v(n * d);
    for (std::size_t i = 0; i < n; ++i) {
        const float* c = centers.data() + pick(rng) * d;
        for (std::size_t j = 0; j < d; ++j) v[i * d + j] = c[j] + g(rng);
    }
    return v;
}

std::vector<Hit> brute(const float* db, std::size_t n, std::size_t d, const float* q, std::size_t k, Metric m) {
    std::vector<Hit> all(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float* v = db + i * d;
        double s = 0;
        for (std::size_t j = 0; j < d; ++j) s += m == Metric::L2 ? double(q[j] - v[j]) * (q[j] - v[j]) : -double(q[j]) * v[j];
        all[i] = {static_cast<std::int64_t>(i), static_cast<float>(s)};
    }
    std::sort(all.begin(), all.end());
    all.resize(std::min(k, n));
    return all;
}

double recall(const std::vector<Hit>& got, const std::vector<Hit>& truth) {
    std::set<std::int64_t> t;
    for (auto& h : truth) t.insert(h.id);
    std::size_t hit = 0;
    for (auto& h : got) hit += t.count(h.id);
    return truth.empty() ? 1.0 : double(hit) / double(truth.size());
}

std::string tmp_path(const char* tag) {
    return "/tmp/vecsearch_test_" + std::string(tag) + "_" + std::to_string(::getpid()) + ".vsix";
}

}  // namespace

TEST_CASE("SIMD kernels match scalar for every tail length") {
    auto a = random_vectors(1, 300, 1), b = random_vectors(1, 300, 2);
    for (std::size_t d = 0; d <= 300; ++d) {
        const float l2s = detail::l2_sq_scalar(a.data(), b.data(), d);
        const float ips = detail::dot_scalar(a.data(), b.data(), d);
        const float tol = 1e-4f * (1.f + float(d));
        CHECK(std::fabs(l2_sq(a.data(), b.data(), d) - l2s) <= tol);
        CHECK(std::fabs(dot(a.data(), b.data(), d) - ips) <= tol);
        if (detail::have_avx2()) {
            CHECK(std::fabs(detail::l2_sq_avx2(a.data(), b.data(), d) - l2s) <= tol);
            CHECK(std::fabs(detail::dot_avx2(a.data(), b.data(), d) - ips) <= tol);
        }
    }
    MESSAGE("active kernel: " << active_kernel());
}

TEST_CASE("TopK equals full sort, including ties and k > n") {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> s(0, 50);   // many ties
    for (std::size_t k : {0u, 1u, 5u, 64u, 1000u}) {
        std::vector<Hit> all;
        TopK t(k);
        for (int i = 0; i < 500; ++i) {
            Hit h{i, float(s(rng))};
            all.push_back(h);
            t.push(h.id, h.score);
        }
        std::sort(all.begin(), all.end());
        all.resize(std::min<std::size_t>(k, all.size()));
        auto got = t.take_sorted();
        REQUIRE(got.size() == all.size());
        for (std::size_t i = 0; i < got.size(); ++i) {
            CHECK(got[i].id == all[i].id);
            CHECK(got[i].score == all[i].score);
        }
    }
}

TEST_CASE("FlatIndex is exact for L2 and inner product") {
    const std::size_t n = 3000, d = 37, nq = 50, k = 10;   // odd d exercises SIMD tails
    auto db = random_vectors(n, d, 11), qs = random_vectors(nq, d, 12);
    for (Metric m : {Metric::L2, Metric::InnerProduct}) {
        FlatIndex idx(d, m);
        idx.add(db.data(), n / 2);
        idx.add(db.data() + (n / 2) * d, n - n / 2);   // multiple adds keep ids contiguous
        REQUIRE(idx.size() == n);
        auto res = idx.search_batch(qs.data(), nq, k, 4);
        for (std::size_t q = 0; q < nq; ++q) {
            auto truth = brute(db.data(), n, d, qs.data() + q * d, k, m);
            auto row = res.row(q);
            CHECK(recall({row.begin(), row.end()}, truth) == 1.0);
            for (std::size_t i = 1; i < k; ++i) CHECK(row[i - 1].score <= row[i].score);
        }
    }
}

TEST_CASE("batch search is independent of thread count and matches single search") {
    const std::size_t n = 5000, d = 64, nq = 33, k = 7;
    auto db = random_vectors(n, d, 21), qs = random_vectors(nq, d, 22);
    FlatIndex idx(d);
    idx.add(db.data(), n);
    auto r1 = idx.search_batch(qs.data(), nq, k, 1);
    auto r8 = idx.search_batch(qs.data(), nq, k, 8);
    for (std::size_t i = 0; i < r1.hits.size(); ++i) CHECK(r1.hits[i].id == r8.hits[i].id);
    auto one = idx.search(qs.data() + 5 * d, k);
    for (std::size_t i = 0; i < k; ++i) CHECK(one[i].id == r1.row(5)[i].id);
}

TEST_CASE("k larger than index pads with id -1") {
    auto db = random_vectors(3, 8, 1);
    FlatIndex idx(8);
    idx.add(db.data(), 3);
    auto r = idx.search_batch(db.data(), 1, 5);
    CHECK(r.row(0)[0].id == 0);
    CHECK(r.row(0)[2].id >= 0);
    CHECK(r.row(0)[3].id == -1);
    CHECK(r.row(0)[4].id == -1);
    CHECK(idx.search(db.data(), 5).size() == 3);
}

TEST_CASE("save/load round-trips through a zero-copy mmap") {
    const std::size_t n = 1234, d = 24;
    auto db = random_vectors(n, d, 31);
    FlatIndex idx(d, Metric::InnerProduct);
    idx.add(db.data(), n);
    const auto path = tmp_path("rt");
    idx.save(path);

    FlatIndex loaded = FlatIndex::load(path);
    CHECK(loaded.read_only());
    CHECK(loaded.size() == n);
    CHECK(loaded.dim() == d);
    CHECK(loaded.metric() == Metric::InnerProduct);
    CHECK(std::equal(db.begin(), db.end(), loaded.vector(0)));
    auto a = idx.search_batch(db.data(), 20, 5), b = loaded.search_batch(db.data(), 20, 5);
    for (std::size_t i = 0; i < a.hits.size(); ++i) CHECK(a.hits[i].id == b.hits[i].id);
    CHECK_THROWS_AS(loaded.add(db.data(), 1), std::logic_error);

    FlatIndex moved = std::move(loaded);    // mapping survives a move
    CHECK(moved.search(db.data(), 1)[0].id == 0);
    std::remove(path.c_str());
}

TEST_CASE("load rejects corrupt files") {
    const auto path = tmp_path("bad");
    auto write = [&](const std::string& bytes) { std::ofstream(path, std::ios::binary) << bytes; };

    write("tiny");
    CHECK_THROWS(FlatIndex::load(path));
    write(std::string(64, '\0'));
    CHECK_THROWS(FlatIndex::load(path));   // bad magic

    auto db = random_vectors(10, 4, 1);
    FlatIndex idx(4);
    idx.add(db.data(), 10);
    idx.save(path);
    {   // truncate one float: size mismatch must be caught before any read
        std::ifstream in(path, std::ios::binary);
        std::string s((std::istreambuf_iterator<char>(in)), {});
        s.resize(s.size() - 4);
        write(s);
    }
    CHECK_THROWS(FlatIndex::load(path));
    CHECK_THROWS(FlatIndex::load("/nonexistent/dir/x.vsix"));
    std::remove(path.c_str());
}

TEST_CASE("IVF: nprobe == nlist is exact; small nprobe keeps high recall") {
    const std::size_t n = 20000, d = 32, nq = 100, k = 10;
    auto db = clustered(n, d, 64, 41), qs = clustered(nq, d, 64, 41 + 1);
    IvfParams p;
    p.nlist = 64;
    p.train_iters = 10;
    IvfIndex ivf(d, p);
    CHECK_THROWS_AS(ivf.add(db.data(), n), std::logic_error);
    ivf.train(db.data(), n, 4);
    ivf.add(db.data(), n, 4);
    REQUIRE(ivf.size() == n);

    std::size_t total = 0;
    for (std::size_t l = 0; l < ivf.nlist(); ++l) total += ivf.list_size(l);
    CHECK(total == n);

    double r_exact = 0, r_probe8 = 0;
    auto exact = ivf.search_batch(qs.data(), nq, k, p.nlist, 4);
    auto approx = ivf.search_batch(qs.data(), nq, k, 8, 4);
    for (std::size_t q = 0; q < nq; ++q) {
        auto truth = brute(db.data(), n, d, qs.data() + q * d, k, Metric::L2);
        r_exact += recall({exact.row(q).begin(), exact.row(q).end()}, truth);
        r_probe8 += recall({approx.row(q).begin(), approx.row(q).end()}, truth);
    }
    r_exact /= nq;
    r_probe8 /= nq;
    MESSAGE("recall@10 nprobe=64: " << r_exact << "  nprobe=8: " << r_probe8);
    CHECK(r_exact == doctest::Approx(1.0));
    CHECK(r_probe8 >= 0.90);
}

TEST_CASE("IVF input validation and empty-cluster repair") {
    auto db = random_vectors(10, 4, 1);
    IvfParams p;
    p.nlist = 16;
    IvfIndex ivf(4, p);
    CHECK_THROWS_AS(ivf.train(db.data(), 10), std::invalid_argument);   // n < nlist
    CHECK_THROWS_AS(ivf.search(db.data(), 1, 1), std::logic_error);

    // Heavily duplicated data forces empty clusters during Lloyd.
    std::vector<float> dup(400 * 4, 1.0f);
    for (std::size_t i = 0; i < 20; ++i) dup[i * 4] = float(i);
    p.nlist = 8;
    IvfIndex ivf2(4, p);
    ivf2.train(dup.data(), 400);
    ivf2.add(dup.data(), 400);
    auto r = ivf2.search(dup.data() + 5 * 4, 1, 8);
    REQUIRE(r.size() == 1);
    CHECK(r[0].score == doctest::Approx(0.0f));
}
