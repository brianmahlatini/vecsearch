# vecsearch

A C++20 vector similarity search library: exact (flat) and approximate (IVF) k-nearest-neighbour search over float32 embeddings, with runtime-dispatched AVX2/FMA kernels, cache-tiled multithreaded batch search, and zero-copy `mmap` index loading.

```
$ ./build/vs_bench            # 100k x 128-dim, 1000 queries, k=10
l2 kernel scalar   :   56.2 M dist/s    21.6 GFLOP/s
l2 kernel avx2     :  135.8 M dist/s    52.2 GFLOP/s
flat      :      856 QPS  (1.17 ms/query)
ivf np=4  :    43398 QPS  recall@10=0.967  speedup vs flat 50.7x
ivf np=8  :    32941 QPS  recall@10=1.000  speedup vs flat 38.5x
ivf np=16 :    20948 QPS  recall@10=1.000  speedup vs flat 24.5x
```
*Single shared vCPU, GCC 13 `-O3`, Gaussian-mixture data (256 clusters). The "scalar" kernel is already auto-vectorised by GCC at `-O3`; the hand-written AVX2+FMA kernel is still 2.4x faster. Recall is measured against the flat index's exact results. Clustered data is kinder to IVF than uniform noise; on real embeddings expect to need a larger `nprobe` for the same recall.*

## Features

- **FlatIndex**: exact search, L2 or inner product, batch or single query
- **IvfIndex**: k-means++ seeded, Lloyd-trained coarse quantiser with `nprobe` recall/latency control (`nprobe == nlist` is exact)
- **Persistence**: versioned 64-byte header + raw float32 rows; atomic write-then-rename save; `load()` is an `mmap` so opening a multi-GB index is O(1)
- **One portable binary**: AVX2 code is compiled with `target` attributes and chosen at startup, so no `-march=native` and no crash on older CPUs
- Deterministic results: ties broken by id, so output never depends on thread count

## Architecture

```
            query batch
                │  static partition (parallel_for, std::jthread)
      ┌─────────┼─────────┐
   thread 0  thread 1  thread N        each owns a slice of queries
      │
      ├─ query tile (8 queries, heaps stay in L1)
      │     └─ database tile (~256 KiB of rows, stays in L2)
      │          └─ score(q, row) → reject if ≥ heap threshold, else push
      ▼
   bounded max-heap top-k per query  →  SearchResult (nq × k, id -1 padding)

IVF:  query → score vs nlist centroids → partial_sort → scan nprobe inverted lists
      (each list stores its vectors contiguously next to their ids)
```

| Component | Choice | Why |
|---|---|---|
| Distance kernels | AVX2+FMA, two 8-wide accumulators, runtime dispatch | Two independent FMA chains hide the 4-cycle FMA latency; dispatch cost is one perfectly-predicted indirect call |
| Top-k | Bounded max-heap + threshold reject | O(n log k) instead of O(n log n); almost every candidate costs one float compare |
| Flat batch scan | Query tile × database tile | Without tiling each query re-streams the whole database from DRAM; with it, a database tile is reused by every query in the tile while hot in L2 |
| Threading | Static partition over queries | Queries are uniform work, so a shared work queue would only add contention |
| IVF lists | Contiguous `vector<float>` per list | A probe is a linear scan the hardware prefetcher handles perfectly |
| k-means | k-means++ seeding, sub-sampled training, double-precision centroid sums, empty-cluster split | Good partitions with bounded training cost; no drift or dead lists |
| Load | Read-only `mmap` + `MADV_SEQUENTIAL` | No copy, no parse, page cache shared across processes serving the same index |

## Design decisions and trade-offs

- **Flat vs IVF.** Flat is exact and needs no training, but costs O(n·d) per query. IVF cuts that to roughly O(nlist·d + n·nprobe/nlist·d) at the price of recall. A common production setup is IVF for serving and flat for offline ground truth, which is exactly how the benchmark measures recall.
- **Why not HNSW?** Graph indexes give better recall per query cost, but they use a lot more memory per vector, are harder to update and delete from, and are much harder to persist as a flat mmap-able file. IVF keeps the memory layout trivially serialisable and the build parallel; HNSW is the natural next index to add.
- **Score convention.** Both metrics are "smaller is better" (`l2²` and `-dot`), so all search code has one ordering and no branches on metric in the heap.
- **Memory safety of loaded files.** `load()` validates magic, version, metric, `dim > 0` and that `64 + n·dim·4` equals the file size using overflow-safe arithmetic *before* any vector is touched, so a truncated or hostile file throws instead of reading out of bounds.
- **Read-only mapped indexes.** A loaded `FlatIndex` refuses `add()` rather than silently copying the whole file into RAM.
- **Known limits.** One vCPU was available while benchmarking, so thread scaling is not shown. IVF persistence and product quantisation (PQ) for compressed vectors are not implemented yet.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/vs_bench [n] [dim] [nq] [threads]
./build/vs_demo

# AddressSanitizer + UBSan
cmake -S . -B build-asan -DVS_SANITIZE=ON -DCMAKE_BUILD_TYPE=Debug && cmake --build build-asan -j
```

Tests cover: SIMD vs scalar agreement for every tail length 0–300, top-k vs full sort with heavy ties and `k > n`, flat exactness against a double-precision brute force for both metrics, identical results across thread counts, save/load round-trip including moves of the mapped index, rejection of truncated/corrupt files, IVF exactness at `nprobe == nlist` and recall ≥ 0.90 at `nprobe = 8`, and empty-cluster repair on degenerate data. CI runs GCC and Clang, each in Release and ASan+UBSan.

## Usage

```cpp
#include "vecsearch/flat_index.hpp"
#include "vecsearch/ivf_index.hpp"

vs::FlatIndex flat(128);                  // L2 by default
flat.add(vectors, n);                     // row-major float[n * 128]
auto top = flat.search(query, 10);        // std::vector<vs::Hit>{id, score}
flat.save("items.vsix");
auto mapped = vs::FlatIndex::load("items.vsix");   // zero-copy

vs::IvfIndex ivf(128, {.nlist = 1024});
ivf.train(vectors, n);
ivf.add(vectors, n);
auto res = ivf.search_batch(queries, nq, 10, /*nprobe=*/8);
for (auto& h : res.row(0)) { /* h.id, h.score */ }
```

## License

MIT
