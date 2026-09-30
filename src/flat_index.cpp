#include "vecsearch/flat_index.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>

#include "vecsearch/parallel.hpp"

static_assert(std::endian::native == std::endian::little, "on-disk format is little-endian");

namespace vs {
namespace {

constexpr char kMagic[4] = {'V', 'S', 'I', 'X'};
constexpr std::uint32_t kVersion = 1;

struct FileHeader {
    char magic[4];
    std::uint32_t version;
    std::uint32_t metric;
    std::uint32_t reserved;
    std::uint64_t dim;
    std::uint64_t n;
    std::uint8_t pad[32];
};
static_assert(sizeof(FileHeader) == 64, "header must be 64 bytes (keeps vectors cache-line aligned)");

// Rows per database tile: ~256 KiB of vectors, comfortably inside L2 on
// anything from the last decade, while leaving room for query rows + heaps.
std::size_t tile_rows(std::size_t dim) noexcept {
    const std::size_t bytes = 256 * 1024;
    return std::max<std::size_t>(64, bytes / (dim * sizeof(float)));
}
constexpr std::size_t kQueryTile = 8;   // heaps + query rows stay in L1

[[noreturn]] void throw_errno(const std::string& what) {
    throw std::system_error(errno, std::generic_category(), what);
}

void write_all(int fd, const void* p, std::size_t len, const std::string& path) {
    const char* c = static_cast<const char*>(p);
    while (len > 0) {
        ssize_t w = ::write(fd, c, len);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw_errno("write " + path);
        }
        c += w;
        len -= static_cast<std::size_t>(w);
    }
}

void fill_row(TopK& t, std::size_t k, Hit* out) {
    auto hits = t.take_sorted();
    std::copy(hits.begin(), hits.end(), out);
    std::fill(out + hits.size(), out + k, Hit{-1, std::numeric_limits<float>::infinity()});
}

}  // namespace

struct FlatIndex::Mapping {
    void* addr = nullptr;
    std::size_t len = 0;
    ~Mapping() { if (addr) ::munmap(addr, len); }
};

FlatIndex::FlatIndex(std::size_t dim, Metric metric) : dim_(dim), metric_(metric) {
    if (dim == 0) throw std::invalid_argument("FlatIndex: dim must be > 0");
}
FlatIndex::~FlatIndex() = default;
FlatIndex::FlatIndex(FlatIndex&&) noexcept = default;
FlatIndex& FlatIndex::operator=(FlatIndex&&) noexcept = default;

void FlatIndex::add(const float* data, std::size_t n) {
    if (read_only()) throw std::logic_error("FlatIndex: index is memory-mapped read-only");
    if (n == 0) return;
    if (!data) throw std::invalid_argument("FlatIndex::add: null data");
    owned_.insert(owned_.end(), data, data + n * dim_);
    n_ += n;
    base_ = owned_.data();   // may have reallocated
}

std::vector<Hit> FlatIndex::search(const float* query, std::size_t k) const {
    auto r = search_batch(query, 1, k, 1);
    auto row = r.row(0);
    std::vector<Hit> out(row.begin(), row.end());
    while (!out.empty() && out.back().id < 0) out.pop_back();
    return out;
}

SearchResult FlatIndex::search_batch(const float* queries, std::size_t nq, std::size_t k,
                                     unsigned threads) const {
    SearchResult res;
    res.k = k;
    res.nq = nq;
    res.hits.resize(nq * k);
    if (nq == 0 || k == 0) return res;
    if (!queries) throw std::invalid_argument("FlatIndex::search_batch: null queries");

    const std::size_t rows = tile_rows(dim_);
    parallel_for(nq, threads, [&](std::size_t qb, std::size_t qe) {
        std::vector<TopK> heaps;
        heaps.reserve(kQueryTile);
        for (std::size_t q0 = qb; q0 < qe; q0 += kQueryTile) {
            const std::size_t q1 = std::min(qe, q0 + kQueryTile);
            heaps.clear();
            for (std::size_t q = q0; q < q1; ++q) heaps.emplace_back(k);

            for (std::size_t r0 = 0; r0 < n_; r0 += rows) {
                const std::size_t r1 = std::min(n_, r0 + rows);
                for (std::size_t q = q0; q < q1; ++q) {
                    const float* qv = queries + q * dim_;
                    TopK& h = heaps[q - q0];
                    float thr = h.threshold();
                    for (std::size_t r = r0; r < r1; ++r) {
                        const float s = score(metric_, qv, base_ + r * dim_, dim_);
                        if (s < thr) {                 // cheap reject keeps the heap cold
                            h.push(static_cast<std::int64_t>(r), s);
                            thr = h.threshold();
                        }
                    }
                }
            }
            for (std::size_t q = q0; q < q1; ++q) fill_row(heaps[q - q0], k, res.hits.data() + q * k);
        }
    });
    return res;
}

void FlatIndex::save(const std::string& path) const {
    FileHeader h{};
    std::memcpy(h.magic, kMagic, 4);
    h.version = kVersion;
    h.metric = static_cast<std::uint32_t>(metric_);
    h.dim = dim_;
    h.n = n_;

    // Write-then-rename: a crash mid-save never leaves a torn index behind.
    const std::string tmp = path + ".tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) throw_errno("open " + tmp);
    try {
        write_all(fd, &h, sizeof h, tmp);
        write_all(fd, base_, n_ * dim_ * sizeof(float), tmp);
        if (::fsync(fd) != 0) throw_errno("fsync " + tmp);
    } catch (...) {
        ::close(fd);
        ::unlink(tmp.c_str());
        throw;
    }
    if (::close(fd) != 0) { ::unlink(tmp.c_str()); throw_errno("close " + tmp); }
    if (::rename(tmp.c_str(), path.c_str()) != 0) { ::unlink(tmp.c_str()); throw_errno("rename " + path); }
}

FlatIndex FlatIndex::load(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw_errno("open " + path);
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); throw_errno("fstat " + path); }
    const auto len = static_cast<std::size_t>(st.st_size);
    if (len < sizeof(FileHeader)) { ::close(fd); throw std::runtime_error(path + ": too small to be an index"); }

    void* addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);   // the mapping keeps the file alive
    if (addr == MAP_FAILED) throw_errno("mmap " + path);
    auto map = std::make_unique<Mapping>();
    map->addr = addr;
    map->len = len;

    FileHeader h;
    std::memcpy(&h, addr, sizeof h);
    if (std::memcmp(h.magic, kMagic, 4) != 0) throw std::runtime_error(path + ": bad magic");
    if (h.version != kVersion) throw std::runtime_error(path + ": unsupported version " + std::to_string(h.version));
    if (h.metric > 1) throw std::runtime_error(path + ": bad metric");
    if (h.dim == 0) throw std::runtime_error(path + ": dim is 0");
    // Validate size with overflow-safe arithmetic before trusting n/dim.
    const std::uint64_t max_elems = (std::numeric_limits<std::uint64_t>::max() - sizeof h) / sizeof(float);
    if (h.n != 0 && h.dim > max_elems / h.n) throw std::runtime_error(path + ": header overflow");
    if (sizeof h + h.n * h.dim * sizeof(float) != len) throw std::runtime_error(path + ": size mismatch");

    ::madvise(addr, len, MADV_SEQUENTIAL);   // flat scans read front to back
    FlatIndex idx(static_cast<std::size_t>(h.dim), static_cast<Metric>(h.metric));
    idx.n_ = static_cast<std::size_t>(h.n);
    idx.base_ = reinterpret_cast<const float*>(static_cast<const char*>(addr) + sizeof h);
    idx.map_ = std::move(map);
    return idx;
}

}  // namespace vs
