#pragma once
// Exact (brute-force) k-NN index over contiguous row-major float vectors.
//
// Batch search is cache-tiled: each worker owns a slice of queries and walks
// the database in blocks sized to stay resident in L2, scoring every query
// in its slice against the block before moving on. That turns a
// memory-bandwidth-bound scan (every query re-streams the whole database)
// into a compute-bound one.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/topk.hpp"

namespace vs {

struct SearchResult {
    std::size_t k = 0, nq = 0;
    std::vector<Hit> hits;     // nq * k, best first per query; id = -1 pads short results
    [[nodiscard]] std::span<const Hit> row(std::size_t q) const noexcept { return {hits.data() + q * k, k}; }
};

class FlatIndex {
public:
    FlatIndex(std::size_t dim, Metric metric = Metric::L2);
    ~FlatIndex();
    FlatIndex(FlatIndex&&) noexcept;
    FlatIndex& operator=(FlatIndex&&) noexcept;
    FlatIndex(const FlatIndex&) = delete;
    FlatIndex& operator=(const FlatIndex&) = delete;

    // Appends n vectors; they receive ids size()..size()+n-1.
    // Throws std::logic_error on a read-only (mmap-loaded) index.
    void add(const float* data, std::size_t n);

    [[nodiscard]] std::vector<Hit> search(const float* query, std::size_t k) const;
    [[nodiscard]] SearchResult search_batch(const float* queries, std::size_t nq, std::size_t k,
                                            unsigned threads = 0) const;

    // Binary format: 64-byte header ("VSIX", version, metric, dim, n) followed
    // by n*dim little-endian float32. load() maps the file read-only: opening a
    // multi-GB index is O(1) and pages fault in on first touch.
    void save(const std::string& path) const;
    [[nodiscard]] static FlatIndex load(const std::string& path);

    [[nodiscard]] std::size_t size() const noexcept { return n_; }
    [[nodiscard]] std::size_t dim() const noexcept { return dim_; }
    [[nodiscard]] Metric metric() const noexcept { return metric_; }
    [[nodiscard]] bool read_only() const noexcept { return map_ != nullptr; }
    [[nodiscard]] const float* vector(std::size_t i) const noexcept { return base_ + i * dim_; }

private:
    struct Mapping;
    std::size_t dim_;
    Metric metric_;
    std::size_t n_ = 0;
    std::vector<float> owned_;
    std::unique_ptr<Mapping> map_;
    const float* base_ = nullptr;
};

}  // namespace vs
