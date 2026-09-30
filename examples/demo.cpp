// Build an index, persist it, reopen it zero-copy and query.
#include <cstdio>
#include <random>
#include <vector>

#include "vecsearch/flat_index.hpp"
#include "vecsearch/ivf_index.hpp"

int main() {
    constexpr std::size_t n = 20000, d = 64;
    std::mt19937_64 rng(7);
    std::normal_distribution<float> g;
    std::vector<float> data(n * d);
    for (auto& x : data) x = g(rng);

    vs::FlatIndex flat(d);
    flat.add(data.data(), n);
    flat.save("demo.vsix");
    auto mapped = vs::FlatIndex::load("demo.vsix");   // mmap, no copy

    const float* query = data.data() + 42 * d;
    std::printf("flat (mmap) top-3 for vector 42:\n");
    for (auto& h : mapped.search(query, 3)) std::printf("  id=%lld dist=%.3f\n", (long long)h.id, h.score);

    vs::IvfParams p;
    p.nlist = 128;
    vs::IvfIndex ivf(d, p);
    ivf.train(data.data(), n);
    ivf.add(data.data(), n);
    std::printf("ivf nprobe=16 top-3:\n");
    for (auto& h : ivf.search(query, 3, 16)) std::printf("  id=%lld dist=%.3f\n", (long long)h.id, h.score);
    std::remove("demo.vsix");
}
