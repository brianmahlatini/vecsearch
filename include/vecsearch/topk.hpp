#pragma once
// Bounded top-k selection: a max-heap of the k best (smallest) scores seen.
// Each candidate costs one comparison against the current worst in the
// common case; O(n log k) overall instead of O(n log n) for a full sort.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace vs {

struct Hit {
    std::int64_t id;
    float        score;   // smaller is better (see distance.hpp)
    friend bool operator<(const Hit& a, const Hit& b) noexcept {
        return a.score < b.score || (a.score == b.score && a.id < b.id);   // deterministic ties
    }
};

class TopK {
public:
    explicit TopK(std::size_t k) : k_(k) { heap_.reserve(k); }

    [[nodiscard]] float threshold() const noexcept {
        return heap_.size() < k_ ? std::numeric_limits<float>::infinity() : heap_.front().score;
    }

    void push(std::int64_t id, float score) {
        if (k_ == 0) return;
        Hit h{id, score};
        if (heap_.size() < k_) {
            heap_.push_back(h);
            std::push_heap(heap_.begin(), heap_.end());
        } else if (h < heap_.front()) {
            std::pop_heap(heap_.begin(), heap_.end());
            heap_.back() = h;
            std::push_heap(heap_.begin(), heap_.end());
        }
    }

    // Best first. Leaves the object empty.
    std::vector<Hit> take_sorted() {
        std::sort_heap(heap_.begin(), heap_.end());
        return std::move(heap_);
    }

private:
    std::size_t k_;
    std::vector<Hit> heap_;
};

}  // namespace vs
