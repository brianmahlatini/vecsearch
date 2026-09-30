#pragma once
// Inverted-file (IVF) approximate k-NN index.
//
// train() runs k-means++ seeding + Lloyd iterations to find `nlist` coarse
// centroids. add() files each vector into the list of its nearest centroid,
// stored contiguously (vectors + ids side by side) so a probe is a linear,
// prefetch-friendly scan. search() scores the query against all centroids,
// visits the `nprobe` closest lists and keeps a bounded top-k. nprobe is the
// recall/latency dial: nprobe == nlist is exact.
#include <cstddef>
#include <cstdint>
#include <vector>

#include "vecsearch/distance.hpp"
#include "vecsearch/flat_index.hpp"
#include "vecsearch/topk.hpp"

namespace vs {

struct IvfParams {
    std::size_t nlist = 256;
    std::size_t train_iters = 20;
    std::size_t max_train_points_per_list = 256;   // sub-sample large training sets
    std::uint64_t seed = 42;
};

class IvfIndex {
public:
    IvfIndex(std::size_t dim, IvfParams params = {}, Metric metric = Metric::L2);

    // Throws std::invalid_argument if n < nlist.
    void train(const float* data, std::size_t n, unsigned threads = 0);
    void add(const float* data, std::size_t n, unsigned threads = 0);   // requires train()

    [[nodiscard]] std::vector<Hit> search(const float* query, std::size_t k, std::size_t nprobe) const;
    [[nodiscard]] SearchResult search_batch(const float* queries, std::size_t nq, std::size_t k,
                                            std::size_t nprobe, unsigned threads = 0) const;

    [[nodiscard]] bool trained() const noexcept { return !centroids_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return n_; }
    [[nodiscard]] std::size_t nlist() const noexcept { return params_.nlist; }
    [[nodiscard]] std::size_t list_size(std::size_t l) const noexcept { return lists_[l].ids.size(); }
    [[nodiscard]] const float* centroid(std::size_t l) const noexcept { return centroids_.data() + l * dim_; }

private:
    struct List { std::vector<float> vecs; std::vector<std::int64_t> ids; };
    [[nodiscard]] std::size_t nearest_centroid(const float* v) const noexcept;
    void search_into(const float* q, std::size_t k, std::size_t nprobe, Hit* out) const;

    std::size_t dim_;
    IvfParams params_;
    Metric metric_;
    std::size_t n_ = 0;
    std::vector<float> centroids_;
    std::vector<List> lists_;
};

}  // namespace vs
