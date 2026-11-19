// Self-test: compares the solver with an independent brute force (n <= 7),
// with closed-form answers (quadratic forms, Maiorana-McFarland), and with
// itself under every combination of pruning switches, with and without the
// symmetry pruning.
#pragma once
#include "gen.hpp"
#include "solver.hpp"
#include "autlib.hpp"
#include "ptests.hpp"

typedef unsigned __int128 u128;

// A stand-in for the library: answers with the given maps after a number of polls.
static std::vector<AffMap> g_fake_gens;
static int g_fake_delay = 0;
struct FakeJob : AutJob {
    int polls = 0;
    int poll(std::vector<AffMap>& gens, std::string& note) override {
        if (polls++ < g_fake_delay) return 0;
        gens = g_fake_gens;
        note = "fake";
        return 1;
    }
    void stop(std::string& note) override { note = "fake stopped"; }
};
static AutJob* fake_start(const u64*, int, double) { return new FakeJob(); }

static u128 xs128(u128 U, unsigned v) {  // result bit i = U bit (i ^ v), v < 128
    u64 lo = perm6((u64)U, v & 63), hi = perm6((u64)(U >> 64), v & 63);
    if (v & 64) std::swap(lo, hi);
    return ((u128)hi << 64) | lo;
}

// All linear subspaces of F_2^n (n <= 7) as element masks, grouped by dimension.
static const std::vector<std::vector<u128>>& all_subspaces(int n) {
    static std::vector<std::vector<u128>> cache[8];
    std::vector<std::vector<u128>>& L = cache[n];
    if (!L.empty()) return L;
    L.resize(n + 1);
    L[0].push_back(1);
    for (int k = 0; k < n; ++k) {
        std::vector<u128> next;
        for (u128 U : L[k])
            for (unsigned v = 1; v < (1u << n); ++v)
                if (!((U >> v) & 1)) next.push_back(U | xs128(U, v));
        std::sort(next.begin(), next.end());
        next.erase(std::unique(next.begin(), next.end()), next.end());
        L[k + 1] = next;
    }
    return L;
}

// Largest k with f constant on a k-flat, by checking every flat.
static int brute_force(const TT& F, int n) {
    u128 S = F[0];
    if (n == 7) S |= (u128)F[1] << 64;
    const auto& L = all_subspaces(n);
    for (int k = n; k >= 0; --k)
        for (u128 U : L[k])
            for (unsigned a = 0; a < (1u << n); ++a) {
                u128 fl = xs128(U, a);
                u128 in = fl & S;
                if (in == fl || in == 0) return k;
            }
    return -1;
}

// Rank of the symplectic form of a quadratic function (ignores affine part).
static int quad_rank(const TT& F, int n) {
    std::vector<u32> rows(n, 0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            int b = getbit(F.data(), 0) ^ getbit(F.data(), 1u << i) ^ getbit(F.data(), 1u << j) ^
                    getbit(F.data(), (1u << i) | (1u << j));
            if (b) rows[i] |= 1u << j;
        }
    int rank = 0;
    std::vector<u32> bas(n, 0);
    for (int i = 0; i < n; ++i) {
        u32 v = rows[i];
        while (v) {
            int h = msb32(v);
            if (!bas[h]) { bas[h] = v; ++rank; break; }
            v ^= bas[h];
        }
    }
    return rank;
}

static int selftest() {
    int fails = 0, checks = 0;
    Options def;
    def.threads = 1;
    double tlap = now_sec();
    auto lap = [&](const char* what) {
        printf("  %-46s %6.1f s\n", what, now_sec() - tlap);
        fflush(stdout);
        tlap = now_sec();
    };
    auto expect = [&](bool cond, const char* what, int n, u64 seed) {
        ++checks;
        if (!cond) { ++fails; printf("FAIL: %s (n=%d seed=%llu)\n", what, n, (unsigned long long)seed); }
    };
    // 1. primitives against bit-by-bit definitions
    {
        std::mt19937_64 rng(7);
        for (int it = 0; it < 2000; ++it) {
            u64 w = rng();
            unsigned v = rng() & 63;
            u64 r = perm6(w, v);
            bool ok = true;
            for (int i = 0; i < 64; ++i) ok &= ((r >> i) & 1) == ((w >> (i ^ v)) & 1);
            expect(ok, "perm6", 0, it);
        }
        for (int m = 1; m <= 10; ++m)
            for (int it = 0; it < 60; ++it) {
                TT T = random_function(m, 1000 * m + it, 1 + it % 3, 4);
                u32 v = 1 + (u32)(rng() % ((1u << m) - 1));
                int p = msb32(v);
                TT out(nwords(m - 1), 0);
                u64 cnt = quotient(T.data(), m, v, out.data());
                bool ok = true;
                u64 c2 = 0;
                for (u32 y = 0; y < (1u << (m - 1)); ++y) {
                    u32 x = insert_bit0(y, p);
                    bool e = getbit(T.data(), x) && getbit(T.data(), x ^ v);
                    c2 += e;
                    ok &= e == getbit(out.data(), y);
                }
                expect(ok && c2 == cnt, "quotient", m, it);
            }
    }
    for (int n = 2; n <= 20; ++n) expect(gf_primitive(n), "field polynomial not primitive", n, 0);
    lap("primitives (perm6, quotient)");
    // 1b. vectorised helpers against their definitions
    {
        std::mt19937_64 rng(11);
        for (int m = 6; m <= 16; ++m)
            for (int it = 0; it < 40; ++it) {
                TT T = random_function(m, 7000 * m + it, 1, 1u << (1 + it % 10));
                u64 want = 0;
                for (u32 x = 0; x < (1u << m); ++x)
                    if (getbit(T.data(), x)) want |= 1ULL << (x >> (m - 6));
                expect(shadow64(T.data(), m) == want, "shadow64", m, it);
            }
        for (int lw = 0; lw <= 5; ++lw)
            for (int lb = 0; lb <= 9; ++lb)
                for (int it = 0; it < 30; ++it) {
                    size_t W = (size_t)1 << lw, nblk = (size_t)1 << lb, tot = W * nblk, vw = rng() % W;
                    std::vector<u64> A(tot), B(tot), O(tot, ~0ULL), R(tot);
                    for (auto& w : A) w = rng() & rng() & rng() & rng();
                    for (auto& w : B) w = rng() & rng() & rng() & rng();
                    u64 nb = 0;
                    for (size_t blk = 0; blk < nblk; ++blk) {
                        u64 acc = 0;
                        for (size_t i = 0; i < W; ++i) acc |= R[blk * W + i] = A[blk * W + i] & B[blk * W + (i ^ vw)];
                        nb += acc != 0;
                    }
                    bool ok = and_blocks(A.data(), B.data(), O.data(), W, nblk, vw, 0, false) && O == R;
                    for (u64 need : {(u64)1, nb, nb + 1, (u64)nblk}) {
                        bool r = and_blocks(A.data(), B.data(), O.data(), W, nblk, vw, need, true);
                        ok &= r == (nb >= need) && (!r || O == R);
                    }
                    expect(ok, "and_blocks", lw * 10 + lb, it);
                    if (lb >= 6) {  // counting version: 64 buckets of 2^(lb-6) blocks
                        int csh = lb - 6;
                        u32 want[64] = {0}, got[64];
                        for (size_t blk = 0; blk < nblk; ++blk) {
                            u64 acc = 0;
                            for (size_t i = 0; i < W; ++i) acc |= R[blk * W + i];
                            want[blk >> csh] += acc != 0;
                        }
                        bool okc = and_count(A.data(), B.data(), W, vw, 0, got, csh);
                        for (int b = 0; b < 64; ++b) okc &= got[b] == want[b];
                        okc &= and_count(A.data(), B.data(), W, vw, nb, got, csh);
                        okc &= !and_count(A.data(), B.data(), W, vw, nb + 1, got, csh);
                        expect(okc, "and_count", lw * 10 + lb, it);
                    }
                }
        for (int it = 0; it < 3000; ++it) {  // one-word flat test against brute force
            int D = 2 + it % 5;
            u64 X = rng() & (it % 3 ? rng() : ~0ULL) & (D == 6 ? ~0ULL : (1ULL << (1 << D)) - 1);
            TT T(1, X);
            int bf = -1;
            const auto& L = all_subspaces(D);
            for (int k = D; k >= 0 && bf < 0; --k)
                for (u128 U : L[k]) {
                    for (unsigned a = 0; a < (1u << D) && bf < 0; ++a) {
                        u128 fl = xs128(U, a);
                        if ((fl & (u128)X) == fl) bf = k;
                    }
                    if (bf >= 0) break;
                }
            bool ok = true;
            for (int r = 0; r <= D; ++r) ok &= flat64(X, D, r, 0) == (r <= bf);
            expect(ok, "flat64", D, it);
        }
    }
    lap("vectorised helpers, one-word flat test");
    // 2. brute force, all pruning configurations
    for (int n = 1; n <= 7; ++n) {
        int reps = n <= 5 ? 300 : (n == 6 ? 150 : 40);
        for (int it = 0; it < reps; ++it) {
            u64 seed = 100 * n + it;
            TT F;
            switch (it % 5) {
                case 0: F = random_function(n, seed); break;
                case 1: F = random_function(n, seed, 1, 4); break;
                case 2: F = random_function(n, seed, 7, 8); break;
                case 3: F = random_degree(n, 2, seed); break;
                default: F = random_degree(n, 3, seed); break;
            }
            int bf = brute_force(F, n);
            {   // anchored search against brute force over linear subspaces
                u128 S = F[0];
                if (n == 7) S |= (u128)F[1] << 64;
                int c = (int)(S & 1), bfa = 0;
                const auto& L = all_subspaces(n);
                for (int k = n; k > 0 && !bfa; --k)
                    for (u128 U : L[k])
                        if ((U & S) == (c ? U : (u128)0)) { bfa = k; break; }
                TT Sc = F;
                if (!c) {
                    for (auto& w : Sc) w = ~w;
                    if (n < 6) Sc[0] &= (1ULL << (1u << n)) - 1;
                }
                bool ok = true;
                for (int t = 0; t <= n; ++t) {
                    Path p;
                    u32 leaf = 0;
                    Stats st;
                    ok &= decide(Sc.data(), n, t, def, p, leaf, st, true) == (t <= bfa);
                }
                expect(ok, "anchored search mismatch", n, seed);
            }
            for (int cfg = 0; cfg < 8; ++cfg) {
                Options o;
                o.threads = 1;
                o.use_sparse = cfg & 1;
                o.use_shadow = cfg & 2;
                o.use_prefix = cfg & 4;
                Result r = solve(F, n, o);
                expect(r.k == bf, "brute force mismatch", n, seed * 10 + cfg);
            }
        }
    }
    lap("brute force n <= 7, 8 configurations");
    // 3. quadratic functions: k = n - h (unbalanced) or n - h - 1 (balanced), rank = 2h
    for (int n = 4; n <= 13; ++n)
        for (int it = 0; it < 12; ++it) {
            TT F = random_degree(n, 2, 5000 + 31 * n + it);
            u64 wt = popcount_arr(F.data(), nwords(n));
            int h = quad_rank(F, n) / 2;
            int want = (wt == 0 || wt == (1ull << n)) ? n : n - h - (wt == (1ull << (n - 1)) ? 1 : 0);
            Result r = solve(F, n, def);
            expect(r.k == want, "quadratic form", n, it);
        }
    lap("quadratic forms n = 4..13");
    // 4. Maiorana-McFarland bent functions (scrambled): exactly n/2
    for (int n = 4; n <= 14; n += 2)
        for (int it = 0; it < 4; ++it) {
            TT F = scramble(mm_bent(n, 77 + it), n, 99 + it);
            Result r = solve(F, n, def);
            expect(r.k == n / 2, "MM bent", n, it);
        }
    lap("Maiorana-McFarland n = 4..14");
    // 5. agreement between pruning configurations and thread counts.
    //    n <= 10: all eight; n = 11..13: the four that keep the prefix
    //    restriction; n = 14: shadow test on, dense-only and multi-threaded.
    for (int n = 8; n <= 14; ++n)
        for (int it = 0; it < (n <= 9 ? 12 : n <= 12 ? 4 : 2); ++it) {
            u64 seed = 900 * n + it;
            TT F = it % 3 == 0 ? random_function(n, seed)
                 : it % 3 == 1 ? random_degree(n, 3, seed)
                               : random_function(n, seed, 1, 8);
            Result base = solve(F, n, def);
            for (int cfg = (n <= 10 ? 0 : n <= 13 ? 4 : 6); cfg < 8; ++cfg) {
                Options o;
                o.threads = cfg == 7 ? 4 : 1;
                o.use_sparse = cfg & 1;
                o.use_shadow = cfg & 2;
                o.use_prefix = cfg & 4;
                Result r = solve(F, n, o);
                expect(r.k == base.k, "configuration mismatch", n, seed * 10 + cfg);
            }
        }
    lap("configuration agreement n = 8..14");
    // 6. planted flats: random function forced constant on a random flat
    for (int n = 10; n <= 14; ++n)
        for (int it = 0; it < 3; ++it) {
            int k = n / 2 + it;
            TT F = random_function(n, 4000 + n + it);
            for (u32 x = 0; x < (1u << k); ++x) F[x >> 6] &= ~(1ULL << (x & 63));  // f = 0 on {x < 2^k}
            TT G = scramble(F, n, 5 + it);
            Result r = solve(G, n, def);
            expect(r.k >= k, "planted flat", n, it);
        }
    lap("planted flats n = 10..14");
    // 7. A flat planted inside a single search node (given leading-bit range):
    //    no pruning test may reject the node, and the node search must succeed.
    {
        std::mt19937_64 rng(4242);
        std::atomic<bool> stop(false);
        Options o;
        o.threads = 1;
        for (int it = 0; it < 4000; ++it) {
            int m = 12 + it % 5;                             // 12..16
            int j = 3 + (int)(rng() % 6);                    // 3..8
            int lo = (int)(rng() % (unsigned)(m - j + 1));   // leading bits from [lo, m-1]
            if (it % 2 && m - j >= 2) lo = m - j - (int)(rng() % 3);  // often nearly forced: the quick check applies
            std::vector<int> piv;
            for (int p = lo; p < m; ++p) piv.push_back(p);
            for (size_t i = piv.size() - 1; i > 0; --i) std::swap(piv[i], piv[rng() % (i + 1)]);
            piv.resize(j);
            u32 a = (u32)rng() & ((1u << m) - 1);
            TT T(nwords(m), 0);
            unsigned den = 4u << (it % 7);                   // background density 1/4 .. 1/256
            for (u32 x = 0; x < (1u << m); ++x)
                if (rng() % den == 0) setbit(T, x);
            std::vector<u32> basis;
            for (int p : piv) basis.push_back((1u << p) | ((u32)rng() & ((1u << p) - 1)));
            for (u32 mask = 0; mask < (1u << j); ++mask) {
                u32 x = a;
                for (int i = 0; i < j; ++i)
                    if (mask >> i & 1) x ^= basis[i];
                setbit(T, x);
            }
            Ctx ctx;
            ctx.init(m, &o, &stop);
            int t6 = m - 6, itrue = 0, imin = -1;
            for (int p : piv) itrue += p >= t6;
            bool ok = true;
            if (lo < t6) {
                u32 bc[64] = {0};
                for (u32 q = 0; q < (1u << (m - lo)); ++q) {
                    bool ne = false;
                    for (u32 y = 0; y < (1u << lo) && !ne; ++y) ne = getbit(T.data(), (q << lo) | y);
                    bc[q >> (t6 - lo)] += ne;
                }
                imin = multi_shadow_min(bc, j, t6 - lo);
                ok &= imin >= 0 && imin <= itrue;
            } else {
                ok &= top6_ok(shadow64(T.data(), m), m, j, lo);
            }
            expect(ok, "shadow test rejects a planted flat", m, it);
            expect(!ctx.quick_dead(T.data(), m, j, lo, imin, [](u32) {}), "quick check rejects a planted flat", m, it);
            ctx.buf[0] = T;
            expect(ctx.descend(0, m, j, lo, popcount_arr(T.data(), nwords(m)), imin), "node search misses a planted flat", m, it);
        }
    }
    lap("flats planted in single nodes, m = 12..16");
    // 8. A flat planted in a random function must be found by the exhaustive
    //    search itself, with the quick check on and off.
    {
        std::mt19937_64 rng(777);
        for (int n = 14; n <= 16; ++n)
            for (int dt = 0; dt <= 1; ++dt)
                for (int it = 0; it < 3; ++it) {
                    int t = n / 2 + dt;
                    TT S = random_function(n, 31000 + 100 * n + 10 * (dt + 1) + it);  // the set searched
                    std::vector<u32> basis, bas(n, 0);
                    while ((int)basis.size() < t) {
                        u32 v = (u32)rng() & ((1u << n) - 1), w = v;
                        while (w) {
                            int h = msb32(w);
                            if (!bas[h]) { bas[h] = w; basis.push_back(v); break; }
                            w ^= bas[h];
                        }
                    }
                    u32 a = (u32)rng() & ((1u << n) - 1);
                    for (u32 mask = 0; mask < (1u << t); ++mask) {
                        u32 x = a;
                        for (int i = 0; i < t; ++i)
                            if (mask >> i & 1) x ^= basis[i];
                        setbit(S, x);
                    }
                    for (int q = 0; q < 2; ++q) {
                        Options o;
                        o.threads = 4;
                        o.use_quick = q;
                        Path p;
                        u32 leaf = 0;
                        Stats st;
                        expect(decide(S.data(), n, t, o, p, leaf, st), "planted flat missed by the search", n, 10 * it + q);
                    }
                }
    }
    lap("planted flats found by the search, n = 14..16");
    // 9. Symmetry: the linear-map helpers against their definitions.
    {
        std::mt19937_64 rng(4242);
        auto rand_inv = [&](int m) {
            for (;;) {
                LMap M, I;
                for (int i = 0; i < 32; ++i) M.c[i] = i < m ? (u32)rng() & ((1u << m) - 1) : 0;
                if (lm_inverse(M, m, I)) return M;
            }
        };
        for (int m = 2; m <= 20; ++m)
            for (int it = 0; it < 20; ++it) {
                LMap M = rand_inv(m), I;
                expect(lm_inverse(M, m, I) && lm_is_identity(lm_mul(M, I, m), m) && lm_is_identity(lm_mul(I, M, m), m),
                       "lm_inverse", m, it);
                // a map fixing v, and the map it induces on the quotient by v
                LMap Pm = rand_inv(m), N = rand_inv(m), Ni;
                Pm.c[0] = 1;
                if (!lm_inverse(Pm, m, I)) continue;
                lm_inverse(N, m, Ni);
                LMap F = lm_mul(N, lm_mul(Pm, Ni, m), m);
                u32 v = N.c[0];
                int p = msb32(v);
                LMap Q = lm_induced(F, m, v, p);
                bool ok = lm_apply(F, v) == v;
                auto proj = [&](u32 x) { if ((x >> p) & 1) x ^= v; return remove_bit(x, p); };
                for (int r = 0; r < 50 && ok; ++r) {
                    u32 x = (u32)rng() & ((1u << m) - 1);
                    ok = proj(lm_apply(F, x)) == lm_apply(Q, proj(x));
                }
                expect(ok, "lm_induced", m, it);
            }
        for (int m = 1; m <= 13; ++m)
            for (int it = 0; it < 30; ++it) {
                TT T = random_function(m, 555 * m + it, 1, 2u << (it % 6));
                u32 v = 1 + (u32)(rng() % ((1u << m) - 1));
                int p = msb32(v);
                TT out(nwords(m - 1), 0);
                or_quotient(T.data(), m, v, out.data());
                bool ok = true;
                for (u32 y = 0; y < (1u << (m - 1)); ++y) {
                    u32 x = insert_bit0(y, p);
                    ok &= (getbit(T.data(), x) || getbit(T.data(), x ^ v)) == getbit(out.data(), y);
                }
                expect(ok, "or_quotient", m, it);
                std::vector<u32> a((size_t)1 << m);
                for (u32& x : a) x = (u32)rng() >> (it % 3 ? 20 : 0);
                u32 cut = (u32)rng() >> (it % 3 ? 20 : 0);
                TT mk(nwords(m), it & 1 ? ~0ULL : 0), mk0 = mk;
                below_mask(a.data(), a.size(), cut, mk.data(), it & 2);
                ok = true;
                for (u32 x = 0; x < (1u << m); ++x)
                    ok &= getbit(mk.data(), x) == ((a[x] < cut) || ((it & 2) && getbit(mk0.data(), x)));
                expect(ok, "below_mask", m, it);
            }
    }
    // Random small groups of linear maps: stabilisers, skip sets and orbit minima
    // against the list of all group elements.
    auto rand_simple = [](std::mt19937_64& rng, int m) {  // a permutation of coordinates or a transvection
        LMap M = lm_identity(m);
        if (rng() % 3) {
            int a = rng() % m, b = rng() % m, c = rng() % m;
            if (a != b) std::swap(M.c[a], M.c[b]);
            if ((rng() & 1) && b != c) std::swap(M.c[b], M.c[c]);
        } else {
            for (;;) {
                u32 a = (u32)rng() & ((1u << m) - 1), b = (u32)rng() & ((1u << m) - 1);
                if (!a || !b || (__builtin_popcount(a & b) & 1)) continue;
                for (int i = 0; i < m; ++i)
                    if ((a >> i) & 1) M.c[i] ^= b;
                break;
            }
        }
        return M;
    };
    {
        std::mt19937_64 rng(99);
        int complete = 0, trials = 0;
        for (int it = 0; it < 150; ++it) {
            int m = 6 + it % 5;
            std::vector<LMap> gens, all;
            for (int g = 0; g < 1 + it % 3; ++g) gens.push_back(rand_simple(rng, m));
            lm_dedupe(gens, m);
            gens.erase(std::remove_if(gens.begin(), gens.end(), [m](const LMap& g) { return lm_is_identity(g, m); }), gens.end());
            if (gens.empty() || !lm_close(gens, m, 20000, all)) continue;
            SymWork w;
            u32 v = 1 + (u32)(rng() % ((1u << m) - 1));
            int p = msb32(v);
            std::vector<LMap> want, a, b;  // the stabiliser of v, induced on the quotient
            for (const LMap& g : all)
                if (lm_apply(g, v) == v) {
                    LMap h = lm_induced(g, m, v, p);
                    if (!lm_is_identity(h, m - 1)) want.push_back(h);
                }
            lm_dedupe(want, m - 1);
            bool fa = sym_child(all, true, m, v, p, a, w);
            bool same = fa && a.size() == want.size();
            for (size_t i = 0; i < a.size() && same; ++i) same = lm_equal(a[i], want[i], m - 1);
            expect(same, "stabiliser from the full list", m, it);
            bool fb = sym_child(gens, false, m, v, p, b, w);
            bool sound = true;
            for (const LMap& h : b) {
                bool in = false;
                for (const LMap& x : want) in |= lm_equal(h, x, m - 1);
                sound &= in;
            }
            expect(sound, "stabiliser from generators contains a foreign map", m, it);
            std::vector<LMap> cl;
            ++trials;
            if (b.empty() ? want.empty() : (lm_close(b, m - 1, 20000, cl) && cl.size() == want.size())) ++complete;
            if (fb) {
                bool eq = b.size() == want.size();
                expect(eq, "stabiliser listed as complete is not", m, it);
            }
            // skip sets: sound always, exact for the full list and for the search over generators
            int lo = (int)(rng() % 3);
            u32 vend = 1u << (m - (int)(rng() % 3));
            std::vector<char> truth(vend, 0);
            for (u32 x = 1; x < vend; ++x)
                for (const LMap& g : all)
                    if (lm_apply(g, x) < x) { truth[x] = 1; break; }
            std::vector<u64> sk;
            for (int mode = 0; mode < 3; ++mode) {
                sym_skip(mode == 0 ? all : gens, mode == 0, mode == 1, m, lo, vend, sk, w);
                bool ok = true;
                for (u32 x = 1u << lo; x < vend; ++x) {
                    bool sbit = getbit(sk.data(), x);
                    ok &= mode == 2 ? (!sbit || truth[x]) : (sbit == (bool)truth[x]);
                }
                expect(ok, "skip set", m, 10 * it + mode);
            }
            std::vector<u32> km;
            sym_orbit_minima(gens, m, km, w);
            bool ok = true;
            for (u32 x = 0; x < (1u << m); ++x) {
                u32 mn = x;
                for (const LMap& g : all) mn = std::min(mn, lm_apply(g, x));
                ok &= km[x] == mn;
            }
            expect(ok, "orbit minima", m, it);
            // the element read off an orbit's spanning tree takes a point to the
            // smallest point of its orbit, and belongs to the group
            SymGroup SG;
            SG.m = m;
            SG.gens = gens;
            sym_finish(SG);
            ok = SG.can_transport();
            for (int r = 0; r < 6 && ok; ++r) {
                u32 x = (u32)rng() & ((1u << m) - 1);
                LMap T;
                sym_transport(SG, x, T);
                bool in = lm_is_identity(T, m);
                for (const LMap& g : all) in |= lm_equal(T, g, m);
                ok = in && lm_apply(T, x) == SG.omin[x] && SG.omin[x] == km[x];
            }
            expect(ok, "transporter to the orbit minimum", m, it);
        }
        expect(complete * 10 >= trials * 9, "stabilisers from random elements are too often incomplete", complete, trials);
    }
    lap("symmetry helpers, stabilisers, skip sets");
    // 9b. Sets invariant under a random group of affine maps, with a planted
    //     flat: the search with symmetry must answer exactly as without, in
    //     every mode (full list / generators only, one thread / several with
    //     the two-pass schedule, given / adapted coordinates).
    {
        std::mt19937_64 rng(31337);
        for (int it = 0; it < 220; ++it) {
            int m = 9 + it % 5, tp = 4 + (int)(rng() % 4);
            u32 mask = (1u << m) - 1;
            std::vector<LMap> lin;
            std::vector<u32> tr;
            for (int g = 0; g < 1 + it % 3; ++g) {
                lin.push_back(rand_simple(rng, m));
                tr.push_back(rng() % 3 ? 0 : (u32)rng() & mask);
            }
            // point orbits of the affine maps
            std::vector<u32> id((size_t)1 << m, ~0u), q;
            u32 norb = 0;
            for (u32 s0 = 0; s0 <= mask; ++s0) {
                if (id[s0] != ~0u) continue;
                q.assign(1, s0);
                id[s0] = norb;
                for (size_t i = 0; i < q.size(); ++i)
                    for (size_t g = 0; g < lin.size(); ++g) {
                        u32 y = lm_apply(lin[g], q[i]) ^ tr[g];
                        if (id[y] == ~0u) { id[y] = norb; q.push_back(y); }
                    }
                ++norb;
            }
            std::vector<char> in(norb, 0);
            unsigned den = 2 + it % 4;
            for (u32 o = 0; o < norb; ++o) in[o] = rng() % den == 0;
            std::vector<u32> basis, bas(m, 0);
            while ((int)basis.size() < tp) {
                u32 v = (u32)rng() & mask, w = v;
                while (w) {
                    int h = msb32(w);
                    if (!bas[h]) { bas[h] = w; basis.push_back(v); break; }
                    w ^= bas[h];
                }
            }
            u32 a0 = (u32)rng() & mask;
            for (u32 c = 0; c < (1u << tp); ++c) {
                u32 x = a0;
                for (int i = 0; i < tp; ++i)
                    if (c >> i & 1) x ^= basis[i];
                in[id[x]] = 1;
            }
            std::vector<u64> SS[2];
            SS[0].assign(nwords(m), 0);
            SS[1].assign(nwords(m), 0);
            for (u32 x = 0; x <= mask; ++x) SS[in[id[x]] ? 1 : 0][x >> 6] |= 1ULL << (x & 63);
            const u64* S = SS[1].data();
            SymGroup G, Gg;
            G.m = Gg.m = m;
            G.gens = Gg.gens = lin;
            sym_finish(G);
            sym_finish(Gg, 0);  // generators only
            if (G.trivial()) continue;
            Recoord rc;
            rc.build(SS, m, sym_adapted_basis(G.gens, m));
            SymGroup Ga = sym_conjugate(G, rc.col);
            for (int t = tp; t <= tp + 1; ++t) {
                Options o;
                o.threads = 1;
                if (it % 7 == 3) o.use_prefix = false;
                if (it % 7 == 5) o.use_shadow = false;
                if (it % 11 == 4) o.use_sparse = false;
                Path p0;
                u32 l0 = 0;
                Stats s0;
                bool plain = decide(S, m, t, o, p0, l0, s0);
                if (t == tp) expect(plain, "planted flat missed without symmetry", m, it);
                for (int mode = 0; mode < 7; ++mode) {
                    Options om = o;
                    if (mode == 2) om.threads = 4;
                    if (mode == 6) om.sym_mates = false;  // the whole group for the first direction only
                    // modes 4, 5: a tiny allowance, so that every root and every second
                    // direction is put aside and searched in slices (passes 2 and 3)
                    if (mode >= 4) { om.threads = 3; om.sym_allowance = mode == 4 ? 1 : 3000; }
                    const SymGroup* g = mode == 1 || mode == 5 ? &Gg : mode == 3 ? &Ga : &G;
                    const u64* Sx = mode == 3 ? rc.S[1].data() : S;
                    Path p1;
                    u32 l1 = 0;
                    Stats s1;
                    bool r = decide(Sx, m, t, om, p1, l1, s1, false, 0, nullptr, nullptr, 1, nullptr, g);
                    expect(r == plain, "symmetry changes the answer", m, 100 * it + 10 * (t - tp) + mode);
                    if (r) {
                        Flat fl;
                        lift(p1, l1, fl);
                        expect(fl.dim() == t && constant_on_flat(Sx, fl.a, fl.basis, 1), "flat found with symmetry is wrong", m,
                               100 * it + mode);
                    }
                }
            }
        }
    }
    lap("symmetry: invariant sets with planted flats");
    // 9c. The whole solver with known symmetries, against itself without them
    //     and against the brute force.
    {
        g_known_sym = true;
        std::vector<Spec> specs;
        for (int n = 5; n <= 12; ++n) {
            specs.push_back(sp_symmetric(threshold_function(n, n / 2 + 1), n));
            specs.push_back(sp_symmetric(threshold_function(n, n / 3 + 1), n));
            specs.push_back(sp_scramble(sp_symmetric(threshold_function(n, n / 2), n), 40 + n));
            specs.push_back(sp_field(power_function(n, (1ull << n) - 2, 0), n));
            specs.push_back(sp_field(power_function(n, 3, 1), n));
            specs.push_back(sp_scramble(sp_field(power_function(n, 5, 0), n), 60 + n));
        }
        for (int n : {6, 8, 10}) {
            specs.push_back(sp_field(kasami(n), n));
            specs.push_back(sp_hyp(sp_field(kasami(n), n), 1));
            specs.push_back(sp_hyp(sp_plus(sp_field(kasami(n), n), 1), 2));
            specs.push_back(sp_scramble(sp_hyp(sp_field(kasami(n), n), 1), 7));
            specs.push_back(sp_dual(sp_field(kasami(n), n)));
            specs.push_back(sp_scramble(sp_ps(ps_ap(n, 3), n), 9));
        }
        specs.push_back(sp_hyp(sp_field(power_function(9, 254, 0), 9), 2));
        specs.push_back(sp_hyp(sp_symmetric(threshold_function(7, 4), 7), 2));
        g_known_sym = false;
        int idx = 0, used = 0;
        for (const Spec& sp : specs) {
            ++idx;
            if (sp.F.empty()) continue;
            Options o = def;
            Result plain = solve(sp.F, sp.n, o);
            for (int mode = 0; mode < 4; ++mode) {
                Options os = def;
                os.sym_cands = &sp.sym;
                if (mode == 1) os.threads = 4;
                if (mode == 2) os.sym_coords = false;
                if (mode == 3) { os.threads = 3; os.sym_allowance = 2000; }
                Result r = solve(sp.F, sp.n, os);
                expect(r.k == plain.k, "solver with known symmetries disagrees", sp.n, 10 * idx + mode);
                used += r.sym_gens > 0 || r.sym_flip;
            }
            if (sp.n <= 7) expect(plain.k == brute_force(sp.F, sp.n), "brute force (symmetric family)", sp.n, idx);
        }
        expect(used > 0, "known symmetries were never used", 0, 0);
        // f = g(x_1..x_(n-1)) + x_0: the translation by e_0 maps f to f + 1
        std::mt19937_64 rng(5);
        for (int n = 6; n <= 11; ++n) {
            TT G0 = random_function(n - 1, 900 + n), F(nwords(n), 0);
            for (u32 x = 0; x < (1u << n); ++x)
                if (getbit(G0.data(), x >> 1) ^ (x & 1)) setbit(F, x);
            std::vector<AffMap> c(1, identity_map(n));
            c[0].t = 1;
            Options os = def;
            os.sym_cands = &c;
            Result r = solve(F, n, os), plain = solve(F, n, def);
            expect(r.k == plain.k, "solver with f ~ f+1 disagrees", n, 0);
            expect(!r.sym_loaded || r.sym_flip, "translation not recognised as f -> f+1", n, 0);
            // maps that are not automorphisms must be thrown out
            std::vector<AffMap> bad;
            for (int i = 0; i < 6; ++i) {
                AffMap a = identity_map(n);
                for (u32& col : a.col) col = (u32)rng() & ((1u << n) - 1);
                a.t = (u32)rng() & ((1u << n) - 1);
                bad.push_back(a);
            }
            TT R0 = random_function(n, 950 + n);
            os.sym_cands = &bad;
            Result rb = solve(R0, n, os), pb = solve(R0, n, def);
            expect(rb.k == pb.k && rb.sym_gens == 0 && !rb.sym_flip, "non-automorphisms were not rejected", n, 0);
        }
    }
    lap("symmetry: solver with known symmetries");
    // 9d. A group that arrives while the searches are under way (as from the
    //     library): at once, after a few polls, or too late to matter.
    {
        g_known_sym = true;
        std::vector<Spec> specs;
        specs.push_back(sp_symmetric(threshold_function(13, 7), 13));
        specs.push_back(sp_symmetric(threshold_function(12, 5), 12));
        specs.push_back(sp_field(power_function(11, (1ull << 11) - 2, 0), 11));
        specs.push_back(sp_field(power_function(13, (1ull << 13) - 2, 0), 13));
        specs.push_back(sp_hyp(sp_plus(sp_field(kasami(10), 10), 1), 2));
        specs.push_back(sp_scramble(sp_hyp(sp_field(kasami(8), 8), 2), 3));
        g_known_sym = false;
        int idx = 0, used = 0;
        for (const Spec& sp : specs) {
            ++idx;
            Result plain = solve(sp.F, sp.n, def);
            for (int delay : {0, 1, 2, 5, 1000000}) {
                g_fake_gens = sp.sym;
                g_fake_delay = delay;
                Options os = def;
                os.threads = delay % 2 ? 4 : 1;
                os.aut_source = fake_start;
                os.aut_budget = 10;
                Result r = solve(sp.F, sp.n, os);
                expect(r.k == plain.k, "solver with a group arriving on the way disagrees", sp.n, 100 * idx + delay % 100);
                if (delay == 0 && plain.searches > 0) {
                    expect(r.sym_gens > 0 || r.sym_flip, "a group available at once was not used", sp.n, idx);
                    used += r.sym_gens > 0;
                }
            }
            // the library is only started after the searches have taken a while
            g_fake_gens = sp.sym;
            g_fake_delay = 0;
            Options os = def;
            os.aut_source = fake_start;
            os.aut_budget = 10;
            os.aut_delay = 1e6;  // never
            Result r = solve(sp.F, sp.n, os);
            expect(r.k == plain.k && !r.sym_loaded && r.sym_gens == 0, "library started before it was due", sp.n, idx);
            os.aut_delay = 0.002;
            os.threads = 2;
            r = solve(sp.F, sp.n, os);
            expect(r.k == plain.k, "solver with a library started late disagrees", sp.n, idx);
        }
        expect(used > 0, "no group from the stand-in library was used", 0, 0);
    }
    lap("symmetry: group arriving during the search");
    // 9e. The external library, if it was built in: its maps are automorphisms,
    //     the answers do not change, and its time limit is enforced.
    if (aut_library_built()) {
        struct LC { TT F; int n; };
        std::vector<LC> lc;
        lc.push_back({threshold_function(9, 5), 9});
        lc.push_back({power_function(9, 510, 0), 9});
        lc.push_back({hwbf(8), 8});
        lc.push_back({random_function(9, 77), 9});
        lc.push_back({add_hyperbolic(power_function(7, 126, 0), 7, 1), 9});
        lc.push_back({power_function(13, (1ull << 13) - 2, 0), 13});
        int li = 0;
        for (const LC& c : lc) {
            ++li;
            std::vector<AffMap> gens;
            std::string note;
            bool ok = aut_library(c.F.data(), c.n, 60.0, gens, note);
            expect(ok, "library gave no answer", c.n, li);
            bool all_ok = true;
            for (const AffMap& a : gens) all_ok &= (int)a.col.size() == c.n && aut_kind(c.F.data(), c.n, to_lmap(a.col), a.t) >= 0;
            expect(all_ok, "library returned a map that is not an automorphism", c.n, li);
            Options os = def;
            os.aut_source = aut_library_start;
            os.aut_budget = 60.0;
            Result r = solve(c.F, c.n, os), plain = solve(c.F, c.n, def);
            expect(r.k == plain.k, "solver with the library running disagrees", c.n, li);
        }
        std::vector<AffMap> gens;
        std::string note;
        double t0 = now_sec();
        bool ok = aut_library(kasami(10).data(), 10, 0.3, gens, note);  // takes the library about 15 s
        expect(!ok && now_sec() - t0 < 8.0, "library time limit not enforced", 10, 0);
        lap("symmetry: external library");
    } else printf("  (external equivalence library not built in: its tests are skipped)\n");
    fails += pary_selftest(checks);
    printf("selftest: %d checks, %d failures\n", checks, fails);
    return fails;
}
