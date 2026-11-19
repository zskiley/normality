// Exhaustive search: "does T (a subset of F_2^m) contain a j-flat?"
//
// The direction space E of a flat has a unique reduced echelon basis
// v_1,...,v_j with increasing leading bits.  We pick the basis in that order.
// After picking v (leading bit p) we pass to the quotient F_2^m/<v>: coordinate
// p is deleted and the new set holds the cosets {x, x^v} lying inside T.  One
// bitset therefore tracks every translate of the partial direction space at
// once, it halves at every level, and every flat is reached exactly once.
//
// Node = (T, m, j, lo): find a j-flat in T whose directions all have leading
// bit >= lo.  Pruning:
//   prefix   the next (lowest) leading bit is at most m-j;
//   shadow   dropping the low lo coordinates is injective on such a flat, so
//            the set of non-empty 2^lo-blocks must itself contain a j-flat;
//            coarser projections give the same kind of condition on the
//            buckets that hold enough non-empty blocks;
//   quota    the pairs {x, x^v} of a viable direction v must occur in a
//            minimum number of buckets, which a few buckets can already rule out;
//   sparse   small sets are handled as point lists by pair differences;
//   symmetry with a group of automorphisms of the set (group.hpp), a candidate
//            direction is skipped if its coset modulo the directions chosen
//            so far contains a vector that the group maps below the first
//            direction, or if the stabiliser of those directions maps it to a
//            smaller vector.
//
// On success the chosen directions are in dirv/dirp (each in the coordinates
// of its own level) and `leaf` is one surviving coset; solver.hpp lifts them
// back to the root coordinates.
#pragma once
#include "bitops.hpp"
#include "group.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#ifdef __AVX2__
#include <immintrin.h>
#endif

struct Options {
    bool use_sparse = true;
    bool use_shadow = true;
    bool use_prefix = true;
    bool use_quick = true;   // few-bucket refutation before a candidate is materialised
    double sparse_weight = 8.0;  // cost of one pair op relative to one word op
    int threads = 1;
    bool verbose = false;
    // Symmetry (used by the driver for the exhaustive searches):
    const std::vector<AffMap>* sym_cands = nullptr;  // candidate automorphisms; each is checked before use
    AutStart aut_source = nullptr;                   // computes the automorphism group, e.g. an external library
    double aut_budget = 0;                           // seconds it may take; 0 = do not use it
    double aut_delay = 0;                            // seconds of exhaustive search before it is started
    bool sym_coords = true;                          // also try coordinates adapted to the group
    bool sym_mates = true;                           // compare with the images that also start with the first direction
    u64 sym_allowance = (u64)1 << 27;                // work a subtree may take before it is split among threads
};

struct Stats {
    u64 dense_nodes = 0, sparse_nodes = 0, cands = 0;
    u64 sym_skipped = 0;  // candidates skipped because a symmetry maps them to a smaller one
    u64 work = 0;  // rough count of 64-bit word operations
    void add(const Stats& o) {
        dense_nodes += o.dense_nodes; sparse_nodes += o.sparse_nodes; cands += o.cands; work += o.work;
        sym_skipped += o.sym_skipped;
    }
};

// Blockwise out[t] = A[t] & B[t ^ vw] over nblk blocks of W words (vw < W).
// With `early` set, returns whether at least `need` blocks are non-empty,
// giving up (and leaving `out` incomplete) as soon as that has become impossible.
#ifdef __AVX2__
template <int IMM>
static inline bool and_blocks_w4(const u64* A, const u64* B, u64* out, size_t W, size_t nblk, size_t vg,
                                 u64 need, bool early) {
    u64 nb = 0;
    for (size_t blk = 0; blk < nblk; ++blk, A += W, B += W, out += W) {
        __m256i acc = _mm256_setzero_si256();
        for (size_t i = 0; i < W; i += 4) {
            __m256i y = _mm256_loadu_si256((const __m256i*)(B + (i ^ vg)));
            if (IMM != 0xE4) y = _mm256_permute4x64_epi64(y, IMM);
            __m256i x = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(A + i)), y);
            _mm256_storeu_si256((__m256i*)(out + i), x);
            acc = _mm256_or_si256(acc, x);
        }
        if (!_mm256_testz_si256(acc, acc)) ++nb;
        else if (early && nb + (nblk - 1 - blk) < need) return false;
    }
    return !early || nb >= need;
}
#endif
static inline bool and_blocks(const u64* A, const u64* B, u64* out, size_t W, size_t nblk, size_t vw,
                              u64 need, bool early) {
    u64 nb = 0;
#ifdef __AVX2__
    const __m256i z = _mm256_setzero_si256();
    if (W >= 4) {
        size_t vg = vw & ~(size_t)3;
        switch (vw & 3) {
            case 0: return and_blocks_w4<0xE4>(A, B, out, W, nblk, vg, need, early);
            case 1: return and_blocks_w4<0xB1>(A, B, out, W, nblk, vg, need, early);
            case 2: return and_blocks_w4<0x4E>(A, B, out, W, nblk, vg, need, early);
            default: return and_blocks_w4<0x1B>(A, B, out, W, nblk, vg, need, early);
        }
    }
    if (W == 1 && nblk >= 4) {
        for (size_t t = 0; t < nblk; t += 4) {
            __m256i x = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(A + t)),
                                         _mm256_loadu_si256((const __m256i*)(B + t)));
            _mm256_storeu_si256((__m256i*)(out + t), x);
            int e = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, z)));
            nb += 4 - __builtin_popcount(e);
            if (early && nb + (nblk - 4 - t) < need) return false;
        }
        return !early || nb >= need;
    }
    if (W == 2 && nblk >= 2) {
        for (size_t t = 0; t < 2 * nblk; t += 4) {
            __m256i y = _mm256_loadu_si256((const __m256i*)(B + t));
            if (vw) y = _mm256_permute4x64_epi64(y, 0xB1);
            __m256i x = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(A + t)), y);
            _mm256_storeu_si256((__m256i*)(out + t), x);
            int e = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, z)));
            nb += ((e & 3) != 3) + ((e & 12) != 12);
            if (early && nb + (nblk - 2 - t / 2) < need) return false;
        }
        return !early || nb >= need;
    }
#endif
    for (size_t blk = 0; blk < nblk; ++blk, A += W, B += W, out += W) {
        u64 acc = 0;
        for (size_t i = 0; i < W; ++i) {
            u64 x = A[i] & B[i ^ vw];
            out[i] = x;
            acc |= x;
        }
        if (acc) ++nb;
        else if (early && nb + (nblk - 1 - blk) < need) return false;
    }
    return !early || nb >= need;
}

// The same intersection without storing it.  The 64 * 2^csh blocks are taken
// as 64 buckets of 2^csh blocks; bcnt[b] receives the number of non-empty
// blocks of bucket b.  Returns whether at least `need` blocks are non-empty,
// giving up (bcnt incomplete) once that is impossible.
#ifdef __AVX2__
template <int IMM>
static inline bool and_count_w4(const u64* A, const u64* B, size_t W, size_t nbb, size_t vg, u64 need, u32* bcnt) {
    u64 nb = 0;
    for (int b = 0; b < 64; ++b) {
        u32 c = 0;
        for (size_t q = 0; q < nbb; ++q, A += W, B += W) {
            __m256i acc = _mm256_setzero_si256();
            for (size_t i = 0; i < W; i += 4) {
                __m256i y = _mm256_loadu_si256((const __m256i*)(B + (i ^ vg)));
                if (IMM != 0xE4) y = _mm256_permute4x64_epi64(y, IMM);
                acc = _mm256_or_si256(acc, _mm256_and_si256(_mm256_loadu_si256((const __m256i*)(A + i)), y));
            }
            c += !_mm256_testz_si256(acc, acc);
        }
        bcnt[b] = c;
        nb += c;
        if (nb + (u64)(63 - b) * nbb < need) return false;
    }
    return nb >= need;
}
#endif
static inline bool and_count(const u64* A, const u64* B, size_t W, size_t vw, u64 need, u32* bcnt, int csh) {
    const size_t nbb = (size_t)1 << csh;
    u64 nb = 0;
#ifdef __AVX2__
    const __m256i z = _mm256_setzero_si256();
    if (W >= 4) {
        size_t vg = vw & ~(size_t)3;
        switch (vw & 3) {
            case 0: return and_count_w4<0xE4>(A, B, W, nbb, vg, need, bcnt);
            case 1: return and_count_w4<0xB1>(A, B, W, nbb, vg, need, bcnt);
            case 2: return and_count_w4<0x4E>(A, B, W, nbb, vg, need, bcnt);
            default: return and_count_w4<0x1B>(A, B, W, nbb, vg, need, bcnt);
        }
    }
    if (W == 1 && nbb >= 4) {
        for (int b = 0; b < 64; ++b) {
            u32 c = 0;
            for (size_t q = 0; q < nbb; q += 4, A += 4, B += 4) {
                __m256i x = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)A), _mm256_loadu_si256((const __m256i*)B));
                c += 4 - __builtin_popcount(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, z))));
            }
            bcnt[b] = c;
            nb += c;
            if (nb + (u64)(63 - b) * nbb < need) return false;
        }
        return nb >= need;
    }
    if (W == 2 && nbb >= 2) {
        for (int b = 0; b < 64; ++b) {
            u32 c = 0;
            for (size_t q = 0; q < nbb; q += 2, A += 4, B += 4) {
                __m256i y = _mm256_loadu_si256((const __m256i*)B);
                if (vw) y = _mm256_permute4x64_epi64(y, 0xB1);
                __m256i x = _mm256_and_si256(_mm256_loadu_si256((const __m256i*)A), y);
                int e = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(x, z)));
                c += ((e & 3) != 3) + ((e & 12) != 12);
            }
            bcnt[b] = c;
            nb += c;
            if (nb + (u64)(63 - b) * nbb < need) return false;
        }
        return nb >= need;
    }
#endif
    for (int b = 0; b < 64; ++b) {
        u32 c = 0;
        for (size_t q = 0; q < nbb; ++q, A += W, B += W) {
            u64 acc = 0;
            for (size_t i = 0; i < W; ++i) acc |= A[i] & B[i ^ vw];
            c += acc != 0;
        }
        bcnt[b] = c;
        nb += c;
        if (nb + (u64)(63 - b) * nbb < need) return false;
    }
    return nb >= need;
}

// The stabiliser of a root direction and its orbit minima, worked out once
// and shared by the threads that split the work below that root.
struct SymShare {
    std::once_flag once;
    std::vector<LMap> gp;
    std::vector<u32> kmin;
    bool full = false, has = false;
    std::atomic<int> left{0};  // tasks that may still use it
    void done() {              // called once per task; frees the data after the last one
        if (left.fetch_sub(1) == 1) {
            std::vector<LMap>().swap(gp);
            std::vector<u32>().swap(kmin);
        }
    }
};

struct Ctx {
    static const int TBL = 16;
    // Per-direction counters of the sparse step: distinct blocks / buckets in
    // which a pair {x, x^v} has been seen (last, lastb: the latest one, +1).
    struct Cell { u32 cnt, last, bc, lastb; };
    struct Scratch { std::vector<u32> list, vs, pass, size, off, cur, child; std::vector<u64> recs; };

    int n0 = 0;
    // Anchored mode: only flats through the origin (linear subspaces).  The
    // origin's coset is point 0 at every level, and a direction v is admissible
    // only if its own coset is still alive, i.e. v itself lies in the set.
    bool anchored = false;
    // Optional work budget (heuristic passes only): once st.work exceeds it the
    // search gives up and `exhausted` is set, so "not found" proves nothing.
    u64 work_limit = ~0ULL;
    bool exhausted = false;
    bool over_budget() {
        if (st.work <= work_limit) return false;
        exhausted = true;
        return true;
    }
    const Options* opt = nullptr;
    std::atomic<bool>* stop = nullptr;   // set when a flat has been found
    std::atomic<bool>* abort = nullptr;  // optional: give up without an answer
    bool stopped() const {
        return stop->load(std::memory_order_relaxed) || (abort && abort->load(std::memory_order_relaxed));
    }
    std::vector<std::vector<u64>> buf, pa, pb;  // per depth
    std::vector<Scratch> ss;
    std::vector<Cell> table;
    std::vector<u64> bm1, bm2;  // bitmaps over directions for quick_dead, zero between uses
    std::vector<u32> qa, qb;
    std::vector<std::vector<u32>> border;  // border[db]: bucket indices of a db-dim space, low weight first
    // Symmetry.  sym is the group for the set being searched, in its
    // coordinates.  low[d] is the set of cosets (points of the space of depth
    // d) containing a vector that the group maps below the first direction.
    // gp[d] holds elements of the stabiliser of the directions chosen above
    // depth d, acting on the space of depth d (all of them if gfull[d], else
    // generators); gskip[d] marks the candidates to skip at a dense node.
    // Down to depth deep_depth the same rule is applied with the stabiliser in
    // place of the group: kmin[d][z] is the smallest point of the orbit of z
    // under the stabiliser at depth d, and a coset z with kmin[d][z] below the
    // direction chosen at depth d is low from there on.  That costs a pass
    // over the whole quotient per node, so it is kept for heavy subtrees.
    // gstop[d]: the stabiliser skipped nothing at depth d, so it is not
    // carried further down.
    // slice_n > 1 restricts the directions tried at depth slice_depth to one
    // class out of slice_n, so that one heavy subtree can be shared among
    // several threads; child_allow > 0 gives every child of that node a work
    // allowance, and those that exceed it are listed in heavy_kids instead of
    // being finished.
    const SymGroup* sym = nullptr;
    std::vector<std::vector<u64>> low;
    int deep_depth = 0;
    static const int SYM_DEEP_DEPTH = 2;
    std::vector<char> gstop;
    std::vector<std::vector<u32>> kmin;
    std::vector<const u32*> kptr;
    std::vector<char> khas;
    std::vector<u64> lowtmp;
    SymShare* share[SYM_DEEP_DEPTH + 1] = {nullptr, nullptr, nullptr};  // per depth, when that node is split
    // Mates: vectors u of the span of the chosen directions that lie in the
    // orbit of the first direction w_1, each with the group element M taking
    // it to w_1.  For every vector x of the span outside {0, u}, the class of
    // M x modulo w_1 (after the stabiliser of w_1 has made it as small as it
    // can, if z1 is set) must not be below the second direction.  es holds
    // the span in root coordinates for the first few levels; mates[d] belongs
    // to the span above depth d, its last newm[d] entries came with the
    // latest direction and still have to mark their forbidden cosets.
    struct Mate { LMap M, Mi; };
    static const int SYM_MATE_DEPTH = 5;
    static const int SYM_MATE_NEW = 3;         // mates taken on per direction, at most
    std::vector<u32> es;
    std::vector<std::vector<Mate>> mates;
    std::vector<int> newm;
    const u32* z1 = nullptr;                   // orbit minima of the stabiliser of w_1 on the first quotient
    u32 slice_n = 0, slice_k = 0;
    int slice_depth = 0;
    u64 child_allow = 0;
    std::vector<u32> heavy_kids;
    static u32 slice_of(u32 v, u32 n) { return (u32)(((u64)(v * 0x9E3779B1u) * n) >> 32); }
    static const int SYM_MIN_DIM = 11;   // below this the nodes are too cheap to be worth it
    static const int SYM_BFS_DEPTH = 1;  // exact orbit minima from generators down to this depth
    std::vector<std::vector<LMap>> gp;
    std::vector<char> gfull;
    std::vector<std::vector<u64>> gskip;
    SymWork sw;
    u32 dirv[34];
    int dirp[34];
    u32 leaf = 0;
    int found_depth = -1;
    Stats st;

    void init(int n, const Options* o, std::atomic<bool>* s) {
        n0 = n; opt = o; stop = s;
        buf.assign(n + 2, {}); pa.assign(n + 2, {}); pb.assign(n + 2, {});
        for (int d = 0; d <= n; ++d) {
            int m = n - d;
            buf[d].assign(nwords(m), 0);
            if (m >= 7) { pa[d].assign(nwords(m) / 2, 0); pb[d].assign(nwords(m) / 2, 0); }
        }
        ss.assign(n + 2, Scratch());
        table.assign((size_t)1 << TBL, Cell{0, 0, 0, 0});
        bm1.assign((size_t)1 << (TBL - 6), 0);
        bm2.assign((size_t)1 << (TBL - 6), 0);
        gp.assign(n + 2, {});
        gfull.assign(n + 2, 0);
        gskip.assign(n + 2, {});
        border.assign(n + 2, {});
        for (int db = 0; db <= n; ++db) {  // 0, the unit vectors, then sums of two
            std::vector<u32>& o = border[db];
            o.push_back(0);
            for (int i = 0; i < db; ++i) o.push_back(1u << i);
            for (int i = 0; i < db && o.size() < 40; ++i)
                for (int k = i + 1; k < db && o.size() < 40; ++k) o.push_back((1u << i) | (1u << k));
        }
    }

    void use_sym(const SymGroup* g) {
        sym = g;
        low.assign(n0 + 2, {});
        for (int d = 0; d <= n0; ++d) low[d].assign(nwords(n0 - d), 0);
        kmin.assign(SYM_DEEP_DEPTH + 1, {});
        kptr.assign(SYM_DEEP_DEPTH + 1, nullptr);
        khas.assign(n0 + 2, 0);
        gstop.assign(n0 + 2, 0);
        lowtmp.assign(nwords(n0), 0);
        es.assign((size_t)2 << SYM_MATE_DEPTH, 0);
        mates.assign(SYM_MATE_DEPTH + 2, {});
        newm.assign(SYM_MATE_DEPTH + 2, 0);
    }
    // After the first direction v: low[1] = the cosets modulo v that meet low(v).
    void root_low(u32 v) {
        below_mask(sym->omin.data(), (size_t)1 << n0, v, low[0].data(), false);
        or_quotient(low[0].data(), n0, v, low[1].data());
        st.work += 2 * nwords(n0);
        es[0] = 0;
        es[1] = v;
        mates[1].clear();
        newm[1] = 0;
        z1 = nullptr;
    }

    // Direction v (in the coordinates of depth d >= 1) is about to be chosen.
    // Extends the span, compares it with its other images that start with
    // w_1, and returns false if one of them has a smaller second direction.
    bool sym_mates(int d, u32 v) {
        if (d > SYM_MATE_DEPTH) return true;
        std::vector<Mate>& out = mates[d + 1];
        newm[d + 1] = 0;
        if (!opt->sym_mates || !sym->can_transport()) { out.clear(); return true; }
        const u32 w1 = dirv[0], w2 = d == 1 ? v : dirv[1];
        const int p1 = dirp[0];
        const u32* om = sym->omin.data();
        u32 u = v;
        for (int k = d - 1; k >= 0; --k) u = insert_bit0(u, dirp[k]);
        const u32 half = (u32)1 << d;
        for (u32 i = 0; i < half; ++i) es[half + i] = es[i] ^ u;
        auto second = [&](u32 y) {  // the class of y modulo w_1, made as small as the stabiliser can
            if ((y >> p1) & 1) y ^= w1;
            u32 z = remove_bit(y, p1);
            return z1 ? z1[z] : z;
        };
        const std::vector<Mate>& in = mates[d];
        for (const Mate& t : in)  // the vectors that come with v, in the views already there
            for (u32 i = half; i < 2 * half; ++i)
                if (second(lm_apply(t.M, es[i])) < w2) return false;
        out = in;
        int added = 0;
        for (u32 i = half; i < 2 * half && added < SYM_MATE_NEW; ++i) {
            if (om[es[i]] != w1) continue;
            Mate t;  // a new view: every vector of the span outside {0, es[i]}
            sym_transport(*sym, es[i], t.M);
            for (u32 k = 1; k < 2 * half; ++k)
                if (k != i && second(lm_apply(t.M, es[k])) < w2) return false;
            lm_inverse(t.M, n0, t.Mi);
            out.push_back(t);
            ++added;
        }
        newm[d + 1] = added;
        st.work += (u64)(in.size() + 1) << (d + 2);
        return true;
    }

    // low[d] has just been made: add the cosets that the mates found with the
    // latest direction forbid, namely those containing a vector x whose image
    // M x has a class below the second direction.
    void sym_forbid(int d) {
        if (d > SYM_MATE_DEPTH + 1 || !newm[d]) return;
        const u32 w2 = dirv[1];
        if (w2 > (1u << 13)) return;
        const int p1 = dirp[0];
        u64* lw = low[d].data();
        const std::vector<Mate>& ms = mates[d];
        for (size_t t = ms.size() - newm[d]; t < ms.size(); ++t)
            for (u32 z = 1; z < w2; ++z) {
                u32 x = lm_apply(ms[t].Mi, insert_bit0(z, p1));
                for (int k = 0; k < d; ++k) {
                    if ((x >> dirp[k]) & 1) x ^= dirv[k];
                    x = remove_bit(x, dirp[k]);
                }
                lw[x >> 6] |= 1ULL << (x & 63);
            }
        st.work += (u64)newm[d] * w2 * 4;
    }

    bool sparse_better(u64 N, int m, int j, int lo) const {
        if (N > (1u << 14)) return false;
        int pmax = opt->use_prefix ? m - j : m - 1;
        if (pmax < lo) return true;
        double dense = (std::ldexp(1.0, pmax + 1) - std::ldexp(1.0, lo)) * std::ldexp(1.0, std::max(m - 7, 0));
        double sp = std::ldexp(1.0, std::max(m - 6, 0)) + (double)N +
                    opt->sparse_weight * (double)N * (double)N / std::ldexp(1.0, opt->use_prefix ? j : 1);
        return sp < dense;
    }

    // Bucket quota.  A j-flat in m dims whose lowest direction v has leading bit
    // p < top = m-j+1 has at least j-(top-p) of its leading bits at or above
    // `top`, so it meets at least 2^(j-(top-p)) of the 2^(j-1) buckets of 2^top
    // points, and v pairs two of its points in each of them.  If the flat is
    // known to have at least imin leading bits among the top six coordinates,
    // that count is at least imin - max(0, 7-j) as well.
    static u32 quota_of(int j, int top, int p, int imin) {
        int e = j - (top - p);
        int f = imin < 0 ? 0 : imin - (j < 7 ? 7 - j : 0);
        int x = e > f ? e : f;
        return x > 0 ? 1u << x : 1;
    }

    // T = buf[d] in m = n0-d dimensions holds cnt points; find a j-flat, pivots >= lo.
    // imin: lower bound on the number of leading bits among the top six
    // coordinates (from the shadow test), or -1 if unknown.
    bool descend(int d, int m, int j, int lo, u64 cnt, int imin = -1) {
        const u64* T = buf[d].data();
        if (j == 0) {
            leaf = (u32)first_set(T, nwords(m));
            found_depth = d;
            return true;
        }
        if (opt->use_sparse && sparse_better(cnt, m, j, lo)) {
            if (slice_n > 1 && d == slice_depth && slice_k != 0) return false;  // not sliced: slice 0 does all of it
            std::vector<u32>& L = ss[d].list;
            L.resize(cnt);
            extract_bits(T, nwords(m), L.data());
            return sparse(d, L.data(), (int)L.size(), m, j, lo, imin);
        }
        return dense(d, T, m, j, lo);
    }

    // Symmetry at a node of depth d >= 1 (space of dimension m): gp[d] = the
    // stabiliser of the direction chosen at depth d-1, as maps on this space,
    // and with `deep` its orbit minima.  If the node is being split among
    // threads, both are worked out once and shared.
    void sym_enter(int d, int m) {
        auto stabiliser = [&](std::vector<LMap>& out, bool& full, std::vector<u32>& km, bool& has) {
            const std::vector<LMap>& PG = d == 1 ? sym->list() : gp[d - 1];
            bool pfull = d == 1 ? sym->full : (bool)gfull[d - 1];
            has = false;
            full = true;
            if (PG.empty() || m < SYM_MIN_DIM || (d >= 2 && gstop[d - 1])) { out.clear(); return; }
            full = sym_child(PG, pfull, m + 1, dirv[d - 1], dirp[d - 1], out, sw);
            if (!out.empty() && d <= deep_depth) {
                sym_orbit_minima(out, m, km, sw);
                has = true;
                st.work += (u64)16 << m;
            }
        };
        khas[d] = 0;
        gstop[d] = 0;
        SymShare* sh = d <= SYM_DEEP_DEPTH ? share[d] : nullptr;
        if (sh) {
            std::call_once(sh->once, [&] { stabiliser(sh->gp, sh->full, sh->kmin, sh->has); });
            gp[d] = sh->gp;
            gfull[d] = sh->full;
            if (sh->has) { kptr[d] = sh->kmin.data(); khas[d] = 1; }
        } else {
            bool full = true, has = false;
            std::vector<u32> none;
            stabiliser(gp[d], full, d <= SYM_DEEP_DEPTH ? kmin[d] : none, has);
            gfull[d] = full;
            if (has) { kptr[d] = kmin[d].data(); khas[d] = 1; }
        }
        if (d == 1) z1 = khas[1] ? kptr[1] : nullptr;
    }

    bool try_child(int d, u32 v, int p, int m, int j, u64 cnt, int imin = -1) {
        dirv[d] = v;
        dirp[d] = p;
        if (sym && d >= 1) {
            if (!sym_mates(d, v)) { ++st.sym_skipped; return false; }
            if (j > 2) {  // the child will choose directions: it needs its low set
                if (khas[d]) {  // plus the cosets the stabiliser maps below v
                    memcpy(lowtmp.data(), low[d].data(), nwords(m) * sizeof(u64));
                    below_mask(kptr[d], (size_t)1 << m, v, lowtmp.data(), true);
                    or_quotient(lowtmp.data(), m, v, low[d + 1].data());
                    st.work += 2 * nwords(m);
                } else or_quotient(low[d].data(), m, v, low[d + 1].data());
                st.work += nwords(m - 1);
                sym_forbid(d + 1);
            }
        }
        return descend(d + 1, m - 1, j - 1, p, cnt, imin);
    }

    // Quick refutation from a few buckets of T (2^top points each, top = m-j+1;
    // the bucket index lives in a space of dimension j-1).  T need not be
    // complete: fetch(b) is called to make bucket b valid before it is read.
    // Let e >= 1 be the number of leading bits of the flat below `top`.  The
    // flat then meets a (j-e)-flat of buckets, and in each of them it is an
    // e-flat with the same directions everywhere.
    //   e = 1: every bucket holds a pair with one common difference;
    //   e = 2: the buckets met form a hyperplane, and each holds a 2-flat, so
    //          one of its pair differences occurs twice.  Buckets without a
    //          repeated difference must then all lie in the other half, which
    //          fails as soon as they affinely span the bucket space.
    // e is at most top-lo, and at most j - (known leading bits at or above top).
    // Returns true only if both cases are excluded.  Sound, not complete.
    template <class Fetch>
    bool quick_dead(const u64* T, int m, int j, int lo, int imin, Fetch fetch) {
        int top = m - j + 1, db = j - 1;
        if (top < 6 || top > TBL || top <= lo) return false;
        int iadj = imin < 0 ? 0 : imin - (j < 7 ? 7 - j : 0);
        int emax = std::min(top - lo, j - iadj);
        if (emax > 2) return false;
        const size_t Wb = (size_t)1 << (top - 6);
        const u64 maxpairs = (u64)2 << (top / 2);
        const std::vector<u32>& ord = border[db];
        std::vector<u32>&cur = qa, &tmp = qb;
        cur.clear();
        u32 pts[96], base = 0, bas[32];
        for (int i = 0; i < 32; ++i) bas[i] = 0;
        int nbad = 0, rank = 0;
        bool alt1 = true, alt2 = emax >= 2, dead = false;
        bool seeded = false;  // cur / bm1 hold the differences common to the buckets used so far
        int used = 0, crowded = 0;
        for (int t = 0; t < (int)ord.size() && t < 32; ++t) {
            if (!alt2 && used >= 8) break;  // a difference common to eight buckets: let the full search look
            u32 b = ord[t];
            fetch(b);
            int np = 0;
            bool big = false;
            for (size_t i = (size_t)b * Wb; i < (size_t)(b + 1) * Wb && !big; ++i)
                for (u64 w = T[i]; w; w &= w - 1) {
                    if (np == 96) { big = true; break; }
                    pts[np++] = (u32)(i * 64 + ctz64(w));
                }
            // A bucket says something only while it has few pairs compared with the
            // number of possible differences.  Crowded ones are skipped (which is
            // sound); a set that is crowded throughout is left to the full search.
            if (big || (u64)np * (np - 1) / 2 > maxpairs) {
                ++crowded;
                if (crowded > 8 || (used == 0 && crowded >= 2)) break;
                continue;
            }
            ++used;
            tmp.clear();
            bool dup = false;
            if (alt2) {  // distinct differences of this bucket, and whether one repeats
                for_pairs(pts, np, top, lo, [&](u32, u32 v) {
                    u64& w = bm2[v >> 6];
                    u64 bit = 1ULL << (v & 63);
                    if (w & bit) dup = true;
                    else { w |= bit; tmp.push_back(v); }
                });
                for (u32 v : tmp) bm2[v >> 6] &= ~(1ULL << (v & 63));
                if (alt1 && seeded) {
                    size_t k = 0;
                    for (u32 v : tmp)
                        if ((bm1[v >> 6] >> (v & 63)) & 1) tmp[k++] = v;
                    tmp.resize(k);
                }
            } else {     // only the differences seen in every bucket so far matter
                for_pairs(pts, np, top, lo, [&](u32, u32 v) {
                    if (!seeded || ((bm1[v >> 6] >> (v & 63)) & 1)) tmp.push_back(v);
                });
            }
            if (alt1) {
                for (u32 v : cur) bm1[v >> 6] &= ~(1ULL << (v & 63));
                for (u32 v : tmp) bm1[v >> 6] |= 1ULL << (v & 63);
                cur.swap(tmp);
                seeded = true;
                if (cur.empty()) alt1 = false;
            }
            if (alt2 && !dup) {
                if (nbad++ == 0) base = b;
                else {
                    u32 v = b ^ base;
                    while (v) {
                        int h = msb32(v);
                        if (!bas[h]) { bas[h] = v; ++rank; break; }
                        v ^= bas[h];
                    }
                }
                if (rank == db) alt2 = false;
            }
            if (!alt1 && !alt2) { dead = true; break; }
        }
        for (u32 v : cur) bm1[v >> 6] &= ~(1ULL << (v & 63));
        return dead;
    }

    bool dense(int d, const u64* T, int m, int j, int lo) {
        ++st.dense_nodes;
        if (over_budget()) return false;
        size_t nw = nwords(m);
        if (j == 1) {  // two points in different 2^lo-blocks
            if (slice_n > 1 && d == slice_depth && slice_k != 0) return false;
            int64_t x0 = first_set(T, nw);
            if (x0 < 0) return false;
            int64_t x1 = first_set(T, nw, (((u64)x0 >> lo) + 1) << lo);
            if (x1 < 0) return false;
            u32 v = (u32)x0 ^ (u32)x1;
            int p = msb32(v);
            dirv[d] = v; dirp[d] = p;
            leaf = remove_bit((u32)x0, p);
            found_depth = d + 1;
            return true;
        }
        int pmax = opt->use_prefix ? m - j : m - 1;
        int jc = j - 1;
        u64 need = (u64)1 << jc;
        u64* out = buf[d + 1].data();
        size_t nwc = nwords(m - 1);
        const bool sh = opt->use_shadow;
        // Symmetry: the stabiliser of the direction just chosen, acting on this
        // quotient, and the candidates it maps to something smaller.
        const u64* skip = nullptr;
        if (sym && d >= 1 && pmax >= lo) {
            sym_enter(d, m);
            skip = low[d].data();
            if (!gp[d].empty()) {
                const u32 vend = (u32)1 << (pmax + 1);
                if (khas[d]) {  // a candidate must be the smallest point of its orbit
                    const u32* km = kptr[d];
                    gskip[d].assign(((size_t)vend + 63) >> 6, 0);
                    for (u32 v = (u32)1 << lo; v < vend; ++v)
                        if (km[v] != v) gskip[d][v >> 6] |= 1ULL << (v & 63);
                } else {
                    sym_skip(gp[d], gfull[d], d <= SYM_BFS_DEPTH, m, lo, vend, gskip[d], sw);
                    st.work += ((u64)gp[d].size() << (pmax + 1 - 3)) + nw;
                    gstop[d] = popcount_arr(gskip[d].data(), gskip[d].size()) == 0;
                }
                for (size_t i = 0; i < gskip[d].size(); ++i) gskip[d][i] |= low[d][i];
                skip = gskip[d].data();
            }
        } else if (sym) { gp[d].clear(); khas[d] = 0; }
        const bool sliced = slice_n > 1 && d == slice_depth;
        // A child of a node that is being split gets a work allowance; if it
        // runs over, nothing is known about it yet and it is listed for later.
        auto go = [&](u32 v, int p, u64 cnt, int imin) -> bool {
            if (!(child_allow && d == slice_depth)) return try_child(d, v, p, m, j, cnt, imin);
            work_limit = st.work + child_allow;
            bool r = try_child(d, v, p, m, j, cnt, imin);
            work_limit = ~0ULL;
            if (!r && exhausted) {
                exhausted = false;
                heavy_kids.push_back(v);
            }
            return r;
        };
        for (int p = lo; p <= pmax; ++p) {
            if (stopped()) return false;
            if (p < 6) {
                for (u32 vlo = 0; vlo < (1u << p); ++vlo) {
                    u32 v = (1u << p) | vlo;
                    if (anchored && !getbit(T, v)) continue;
                    if (sliced && slice_of(v, slice_n) != slice_k) continue;
                    if (skip && ((skip[v >> 6] >> (v & 63)) & 1)) { ++st.sym_skipped; continue; }
                    ++st.cands;
                    st.work += 2 * nw;
                    u64 cnt = quotient(T, m, v, out);
                    if (cnt < need) continue;
                    if (sh && !top6_ok(shadow64(out, m - 1), m - 1, jc, p)) continue;
                    if (go(v, p, cnt, -1)) return true;
                    if (exhausted) return false;
                }
                continue;
            }
            int D = m - 1 - p;  // child has 2^D blocks of 2^p bits
            if (sh && D < jc) break;
            size_t W = (size_t)1 << (p - 6), nblk = (size_t)1 << D, half = W * nblk;
            u64* A = pa[d].data();
            u64* B = pb[d].data();
            for (size_t blk = 0; blk < nblk; ++blk) {
                memcpy(A + blk * W, T + blk * 2 * W, W * sizeof(u64));
                memcpy(B + blk * W, T + blk * 2 * W + W, W * sizeof(u64));
            }
            // With at least 64 blocks the shadow tests need only block counts per
            // bucket, so the candidate is first evaluated without storing it.
            const bool staged = sh && D >= 6;
            const int csh = D - 6;
            // Quick look at a few buckets of the child (2^topc points each).
            const int topc = m - jc;
            const bool quick = staged && opt->use_quick && opt->use_prefix && jc >= 4 && topc >= 6 && topc <= TBL && topc > p;
            const size_t qblk = quick ? (size_t)1 << (topc - p) : 0, qw = qblk * W;  // blocks, words per bucket
            for (unsigned g = 0; g < 64; ++g) {  // low six bits of v in Gray order
                if (g) {
                    int b = __builtin_ctz(g);
                    for (size_t i = 0; i < half; ++i) B[i] = swapb(B[i], b);
                }
                unsigned vb = g ^ (g >> 1);
                st.work += half;
                if (stopped()) return false;
                for (size_t vw = 0; vw < W; ++vw) {
                    if (anchored && !getbit(T, (1u << p) | ((u32)vw << 6) | vb)) continue;
                    if (skip || sliced) {
                        u32 vv = (1u << p) | ((u32)vw << 6) | vb;
                        if (sliced && slice_of(vv, slice_n) != slice_k) continue;
                        if (skip && ((skip[vv >> 6] >> (vv & 63)) & 1)) { ++st.sym_skipped; continue; }
                    }
                    ++st.cands;
                    st.work += nwc;
                    int imin = -1;
                    if (staged) {
                        u32 bc[64];
                        if (!and_count(A, B, W, vw, need, bc, csh)) continue;
                        if (D == 6) {  // the top-six projection is injective on the flat
                            u64 mask = 0;
                            for (int b = 0; b < 64; ++b) mask |= (u64)(bc[b] != 0) << b;
                            if (!top6_ok(mask, m - 1, jc, p)) continue;
                        } else {
                            imin = multi_shadow_min(bc, jc, csh);
                            if (imin < 0) continue;
                        }
                        if (quick && quick_dead(out, m - 1, jc, p, imin, [&](u32 b) {
                                and_blocks(A + b * qw, B + b * qw, out + b * qw, W, qblk, vw, 0, false);
                            }))
                            continue;
                        and_blocks(A, B, out, W, nblk, vw, 0, false);
                    } else {
                        if (!and_blocks(A, B, out, W, nblk, vw, need, sh)) continue;
                        if (sh && !top6_ok(shadow64(out, m - 1), m - 1, jc, p)) continue;
                    }
                    u64 cnt = popcount_arr(out, nwc);
                    if (cnt < need) continue;
                    u32 v = (1u << p) | ((u32)vw << 6) | vb;
                    if (go(v, p, cnt, imin)) return true;
                    if (exhausted) return false;
                }
            }
        }
        return false;
    }

    // Visit the pairs x < y of the sorted list L that agree above bit `top`
    // and differ at or above bit `lo`; calls fn(x, x ^ y).
    template <class Fn> static void for_pairs(const u32* L, int N, int top, int lo, Fn fn) {
        for (int s = 0; s < N;) {
            int e = s + 1;
            u32 key = L[s] >> top;
            while (e < N && (L[e] >> top) == key) ++e;
            int i2 = s;
            for (int i = s; i < e; ++i) {
                u32 x = L[i], g = x >> lo;
                if (i2 <= i) i2 = i + 1;
                while (i2 < e && (L[i2] >> lo) == g) ++i2;
                for (int k = i2; k < e; ++k) fn(x, x ^ L[k]);
            }
            s = e;
        }
    }

    // L sorted ascending, points of F_2^m.  The next direction v (leading bit p
    // in [lo, top)) pairs points inside one bucket of 2^top points.  v is viable
    // only if the pairs {x, x^v} hit at least 2^(j-1) distinct 2^(p+1)-blocks
    // and at least quota_of(p) distinct buckets.  Buckets are processed in
    // order; once no direction with leading bit p can still reach its quota,
    // pairs of that class are no longer enumerated, and the node is abandoned
    // when every class is dead.
    // lowp: low[d] has not been computed yet (the parent was a sparse node); it
    // is derived from low[d-1] only if some direction gets far enough to need it.
    bool sparse(int d, const u32* L, int N, int m, int j, int lo, int imin = -1, bool lowp = false) {
        ++st.sparse_nodes;
        if (N < (1 << j)) return false;
        if (j == 1) {
            if ((L[0] >> lo) == (L[N - 1] >> lo)) return false;
            u32 v = L[0] ^ L[N - 1];
            int p = msb32(v);
            dirv[d] = v; dirp[d] = p;
            leaf = remove_bit(L[0], p);
            found_depth = d + 1;
            return true;
        }
        int top = (opt->use_prefix ? m - j : m - 1) + 1;  // next direction is < 2^top
        if (top <= lo) return false;
        const bool sh = opt->use_shadow;
        if (sh) {
            int distinct = 1;
            for (int i = 1; i < N; ++i) distinct += (L[i] >> lo) != (L[i - 1] >> lo);
            if (distinct < (1 << j)) return false;
        }
        if (stopped() || over_budget()) return false;
        int jc = j - 1;
        u32 need = 1u << jc;
        st.work += (u64)N * (u64)N / ((u64)2 << (opt->use_prefix ? jc : 0)) + N;  // about the number of pairs
        Scratch& S = ss[d];
        if (top <= TBL) {
            const bool quota = sh && opt->use_prefix;  // the quotas rest on both arguments
            const u32 nbuckets = 1u << (m - top);
            u32 bneed[32], mxb[32];
            for (int p = lo; p < top; ++p) {
                bneed[p] = quota ? quota_of(j, top, p, imin) : 1;
                mxb[p] = 0;
            }
            S.vs.clear(); S.pass.clear();
            Cell* tb = table.data();
            int pcap = top - 1;  // largest leading bit whose class is still alive
            for (int s = 0; s < N && pcap >= lo;) {
                u32 key = L[s] >> top;
                int e = s + 1;
                while (e < N && (L[e] >> top) == key) ++e;
                for_pairs(L + s, e - s, pcap + 1, lo, [&](u32 x, u32 v) {
                    Cell& c = tb[v];
                    int p = msb32(v);
                    u32 blk = (x >> (p + 1)) + 1;
                    if (c.lastb == 0) S.vs.push_back(v);  // first touch
                    if (!sh || c.last != blk) { c.last = blk; ++c.cnt; }
                    if (c.lastb != key + 1) {
                        c.lastb = key + 1;
                        if (++c.bc > mxb[p]) mxb[p] = c.bc;
                    }
                });
                if (quota) {
                    u32 rem = nbuckets - 1 - key;  // buckets still to come (empty ones included)
                    while (pcap >= lo && mxb[pcap] + rem < bneed[pcap]) --pcap;
                }
                s = e;
            }
            for (u32 v : S.vs) {
                const Cell& c = tb[v];
                int p = msb32(v);
                if (p <= pcap && c.cnt >= need && c.bc >= bneed[p]) S.pass.push_back(v);
            }
            if (sym && d >= 1 && !S.pass.empty()) {  // symmetry: drop directions whose coset is low
                if (lowp) {
                    or_quotient(low[d - 1].data(), m + 1, dirv[d - 1], low[d].data());
                    st.work += nwords(m);
                    sym_forbid(d);
                }
                const u64* lw = low[d].data();
                size_t k = 0;
                for (u32 v : S.pass) {
                    if ((lw[v >> 6] >> (v & 63)) & 1) ++st.sym_skipped;
                    else S.pass[k++] = v;
                }
                S.pass.resize(k);
            }
            size_t np = S.pass.size();
            if (np == 0) {
                for (u32 v : S.vs) tb[v] = Cell{0, 0, 0, 0};
                return false;
            }
            // collect the cosets of every passing direction (marked by bc = MARK, index in last)
            const u32 MARK = 0xFFFFFFFFu;
            S.size.assign(np, 0);
            for (size_t i = 0; i < np; ++i) { tb[S.pass[i]].last = (u32)i; tb[S.pass[i]].bc = MARK; }
            for_pairs(L, N, pcap + 1, lo, [&](u32, u32 v) {
                const Cell& c = tb[v];
                if (c.bc == MARK) ++S.size[c.last];
            });
            S.off.assign(np + 1, 0);
            for (size_t i = 0; i < np; ++i) S.off[i + 1] = S.off[i] + S.size[i];
            S.child.resize(S.off[np]);
            S.cur.assign(np, 0);
            for_pairs(L, N, pcap + 1, lo, [&](u32 x, u32 v) {
                const Cell& c = tb[v];
                if (c.bc == MARK) S.child[S.off[c.last] + S.cur[c.last]++] = remove_bit(x, msb32(v));
            });
            for (u32 v : S.vs) tb[v] = Cell{0, 0, 0, 0};
            for (size_t i = 0; i < np; ++i) {
                if (anchored && !std::binary_search(L, L + N, S.pass[i])) continue;
                if (sparse_child(d, S.pass[i], S.child.data() + S.off[i], (int)S.size[i], m, jc)) return true;
                if (exhausted) return false;
            }
            return false;
        }
        // large direction range: sort the (v, x) records instead of using the table
        S.recs.clear();
        for_pairs(L, N, top, lo, [&](u32 x, u32 v) { S.recs.push_back((u64)v << 32 | x); });
        std::sort(S.recs.begin(), S.recs.end());
        size_t P = S.recs.size();
        bool lowready = !lowp;
        for (size_t a = 0; a < P;) {
            u32 v = (u32)(S.recs[a] >> 32);
            int p = msb32(v);
            size_t b = a;
            u32 nb = 0, last = 0;
            for (; b < P && (u32)(S.recs[b] >> 32) == v; ++b) {
                u32 blk = ((u32)S.recs[b] >> (p + 1)) + 1;
                if (!sh || blk != last) { last = blk; ++nb; }
            }
            if (nb >= need && sym && d >= 1) {  // symmetry: skip a direction whose coset is low
                if (!lowready) {
                    or_quotient(low[d - 1].data(), m + 1, dirv[d - 1], low[d].data());
                    st.work += nwords(m);
                    sym_forbid(d);
                    lowready = true;
                }
                if (getbit(low[d].data(), v)) { ++st.sym_skipped; a = b; continue; }
            }
            if (nb >= need && !(anchored && !std::binary_search(L, L + N, v))) {
                S.child.resize(b - a);
                for (size_t t = a; t < b; ++t) S.child[t - a] = remove_bit((u32)S.recs[t], p);
                if (sparse_child(d, v, S.child.data(), (int)(b - a), m, jc)) return true;
                if (exhausted) return false;
            }
            a = b;
        }
        return false;
    }

    bool sparse_child(int d, u32 v, const u32* C, int Nc, int m, int jc) {
        int p = msb32(v), mc = m - 1, imin = -1;
        if (opt->use_shadow) {
            int sft = mc > 6 ? mc - 6 : 0;
            if (p >= sft) {  // the top-six projection is injective on the flat
                u64 mask = 0;
                for (int t = 0; t < Nc; ++t) mask |= 1ULL << (C[t] >> sft);
                if (!top6_ok(mask, mc, jc, p)) return false;
            } else {         // count the distinct 2^p-blocks per bucket (C is sorted)
                u32 bc[64] = {0};
                u32 prev = ~0u;
                for (int t = 0; t < Nc; ++t) {
                    u32 blk = C[t] >> p;
                    bc[C[t] >> sft] += blk != prev;
                    prev = blk;
                }
                imin = multi_shadow_min(bc, jc, sft - p);
                if (imin < 0) return false;
            }
        }
        dirv[d] = v;
        dirp[d] = p;
        if (sym && d >= 1 && !sym_mates(d, v)) { ++st.sym_skipped; return false; }
        return sparse(d + 1, C, Nc, m - 1, jc, p, imin, true);
    }
};
