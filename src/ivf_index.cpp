#include "vecsearch/ivf_index.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "vecsearch/parallel.hpp"

namespace vs {

IvfIndex::IvfIndex(std::size_t dim, IvfParams params, Metric metric)
    : dim_(dim), params_(params), metric_(metric) {
    if (dim == 0) throw std::invalid_argument("IvfIndex: dim must be > 0");
    if (params.nlist == 0) throw std::invalid_argument("IvfIndex: nlist must be > 0");
}

std::size_t IvfIndex::nearest_centroid(const float* v) const noexcept {
    std::size_t best = 0;
    float best_s = std::numeric_limits<float>::infinity();
    for (std::size_t c = 0; c < params_.nlist; ++c) {
        const float s = score(metric_, v, centroid(c), dim_);
        if (s < best_s) { best_s = s; best = c; }
    }
    return best;
}

void IvfIndex::train(const float* data, std::size_t n, unsigned threads) {
    const std::size_t K = params_.nlist;
    if (!data) throw std::invalid_argument("IvfIndex::train: null data");
    if (n < K) throw std::invalid_argument("IvfIndex::train: need at least nlist training points");

    std::mt19937_64 rng(params_.seed);

    // Sub-sample: k-means quality saturates around a few hundred points per
    // centroid, and training cost is O(n * K * d * iters).
    std::vector<std::size_t> sample(n);
    std::iota(sample.begin(), sample.end(), std::size_t{0});
    const std::size_t m = std::min(n, K * params_.max_train_points_per_list);
    if (m < n) {
        for (std::size_t i = 0; i < m; ++i) {       // partial Fisher-Yates
            std::uniform_int_distribution<std::size_t> pick(i, n - 1);
            std::swap(sample[i], sample[pick(rng)]);
        }
        sample.resize(m);
    }
    auto pt = [&](std::size_t i) { return data + sample[i] * dim_; };

    // k-means++ seeding (always in L2: it's a geometric partition).
    centroids_.assign(K * dim_, 0.0f);
    std::vector<float> d2(m, std::numeric_limits<float>::infinity());
    std::uniform_int_distribution<std::size_t> first(0, m - 1);
    std::memcpy(centroids_.data(), pt(first(rng)), dim_ * sizeof(float));
    for (std::size_t c = 1; c < K; ++c) {
        const float* prev = centroids_.data() + (c - 1) * dim_;
        double total = 0;
        for (std::size_t i = 0; i < m; ++i) {
            d2[i] = std::min(d2[i], l2_sq(pt(i), prev, dim_));
            total += d2[i];
        }
        std::size_t chosen = m - 1;
        if (total > 0) {
            double r = std::uniform_real_distribution<double>(0, total)(rng);
            for (std::size_t i = 0; i < m; ++i) {
                r -= d2[i];
                if (r <= 0) { chosen = i; break; }
            }
        } else {
            chosen = first(rng);   // all points identical to existing centroids
        }
        std::memcpy(centroids_.data() + c * dim_, pt(chosen), dim_ * sizeof(float));
    }

    // Lloyd iterations. Assignment is the O(m*K*d) hot loop: parallel.
    std::vector<std::uint32_t> assign(m, 0);
    std::vector<double> sums(K * dim_);
    std::vector<std::size_t> counts(K);
    for (std::size_t it = 0; it < params_.train_iters; ++it) {
        parallel_for(m, threads, [&](std::size_t b, std::size_t e) {
            for (std::size_t i = b; i < e; ++i) {
                std::size_t best = 0;
                float bs = std::numeric_limits<float>::infinity();
                for (std::size_t c = 0; c < K; ++c) {
                    const float s = l2_sq(pt(i), centroids_.data() + c * dim_, dim_);
                    if (s < bs) { bs = s; best = c; }
                }
                assign[i] = static_cast<std::uint32_t>(best);
            }
        });

        std::fill(sums.begin(), sums.end(), 0.0);   // double accumulation: no drift on big clusters
        std::fill(counts.begin(), counts.end(), std::size_t{0});
        for (std::size_t i = 0; i < m; ++i) {
            const float* p = pt(i);
            double* s = sums.data() + assign[i] * dim_;
            for (std::size_t j = 0; j < dim_; ++j) s[j] += p[j];
            ++counts[assign[i]];
        }
        for (std::size_t c = 0; c < K; ++c) {
            float* cv = centroids_.data() + c * dim_;
            if (counts[c] > 0) {
                const double inv = 1.0 / static_cast<double>(counts[c]);
                for (std::size_t j = 0; j < dim_; ++j) cv[j] = static_cast<float>(sums[c * dim_ + j] * inv);
                continue;
            }
            // Empty cluster: split the largest one by stealing a random member
            // and nudging it, the standard fix that keeps all K lists useful.
            const auto big = static_cast<std::size_t>(
                std::max_element(counts.begin(), counts.end()) - counts.begin());
            std::vector<std::size_t> members;
            for (std::size_t i = 0; i < m; ++i) if (assign[i] == big) members.push_back(i);
            const std::size_t donor = members[std::uniform_int_distribution<std::size_t>(0, members.size() - 1)(rng)];
            const float* bc = centroids_.data() + big * dim_;
            const float* dp = pt(donor);
            for (std::size_t j = 0; j < dim_; ++j) cv[j] = bc[j] + 0.5f * (dp[j] - bc[j]);
            counts[big] -= 1;
            counts[c] = 1;
            assign[donor] = static_cast<std::uint32_t>(c);
        }
    }
    lists_.assign(K, List{});
    n_ = 0;
}

void IvfIndex::add(const float* data, std::size_t n, unsigned threads) {
    if (!trained()) throw std::logic_error("IvfIndex::add: call train() first");
    if (n == 0) return;
    if (!data) throw std::invalid_argument("IvfIndex::add: null data");

    std::vector<std::uint32_t> assign(n);
    parallel_for(n, threads, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) assign[i] = static_cast<std::uint32_t>(nearest_centroid(data + i * dim_));
    });
    // Size lists exactly once, then fill: avoids repeated reallocation.
    std::vector<std::size_t> add_counts(params_.nlist, 0);
    for (auto a : assign) ++add_counts[a];
    for (std::size_t l = 0; l < params_.nlist; ++l) {
        lists_[l].ids.reserve(lists_[l].ids.size() + add_counts[l]);
        lists_[l].vecs.reserve(lists_[l].vecs.size() + add_counts[l] * dim_);
    }
    for (std::size_t i = 0; i < n; ++i) {
        List& L = lists_[assign[i]];
        L.ids.push_back(static_cast<std::int64_t>(n_ + i));
        L.vecs.insert(L.vecs.end(), data + i * dim_, data + (i + 1) * dim_);
    }
    n_ += n;
}

void IvfIndex::search_into(const float* q, std::size_t k, std::size_t nprobe, Hit* out) const {
    const std::size_t K = params_.nlist;
    nprobe = std::clamp<std::size_t>(nprobe, 1, K);

    // Coarse step: rank centroids, keep the nprobe best (partial sort, O(K log nprobe)).
    std::vector<Hit> coarse(K);
    for (std::size_t c = 0; c < K; ++c) coarse[c] = Hit{static_cast<std::int64_t>(c), score(metric_, q, centroid(c), dim_)};
    std::partial_sort(coarse.begin(), coarse.begin() + static_cast<std::ptrdiff_t>(nprobe), coarse.end());

    TopK top(k);
    for (std::size_t p = 0; p < nprobe; ++p) {
        const List& L = lists_[static_cast<std::size_t>(coarse[p].id)];
        const float* v = L.vecs.data();
        float thr = top.threshold();
        for (std::size_t i = 0; i < L.ids.size(); ++i, v += dim_) {
            const float s = score(metric_, q, v, dim_);
            if (s < thr) { top.push(L.ids[i], s); thr = top.threshold(); }
        }
    }
    auto hits = top.take_sorted();
    std::copy(hits.begin(), hits.end(), out);
    std::fill(out + hits.size(), out + k, Hit{-1, std::numeric_limits<float>::infinity()});
}

std::vector<Hit> IvfIndex::search(const float* query, std::size_t k, std::size_t nprobe) const {
    if (!trained()) throw std::logic_error("IvfIndex::search: index not trained");
    std::vector<Hit> out(k);
    if (k == 0) return out;
    search_into(query, k, nprobe, out.data());
    while (!out.empty() && out.back().id < 0) out.pop_back();
    return out;
}

SearchResult IvfIndex::search_batch(const float* queries, std::size_t nq, std::size_t k,
                                    std::size_t nprobe, unsigned threads) const {
    if (!trained()) throw std::logic_error("IvfIndex::search_batch: index not trained");
    SearchResult res;
    res.k = k;
    res.nq = nq;
    res.hits.resize(nq * k);
    if (nq == 0 || k == 0) return res;
    parallel_for(nq, threads, [&](std::size_t b, std::size_t e) {
        for (std::size_t q = b; q < e; ++q) search_into(queries + q * dim_, k, nprobe, res.hits.data() + q * k);
    });
    return res;
}

}  // namespace vs
