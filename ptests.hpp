// Self-test of the solver for F_p^n -> F_p: primitives against their
// definitions, the solver against a brute force over all flats (tiny cases),
// against closed-form answers (quadratic forms, Maiorana-McFarland), against
// the binary solver for p = 2, and against itself in its search modes.
#pragma once
#include "pgen.hpp"
#include <map>
#include <set>

// All linear subspaces of F_p^n as sorted point lists, by dimension (tiny cases).
static const std::vector<std::vector<std::vector<u32>>>& pa_all_subspaces(int p, int n) {
    static std::map<std::pair<int, int>, std::vector<std::vector<std::vector<u32>>>> cache;
    auto& L = cache[{p, n}];
    if (!L.empty()) return L;
    u32 N = 1;
    for (int i = 0; i < n; ++i) N *= p;
    L.resize(n + 1);
    L[0].push_back({0});
    for (int k = 0; k < n; ++k) {
        std::set<std::vector<u32>> next;
        for (const auto& U : L[k]) {
            std::vector<char> cov(N, 0);
            for (u32 u : U) cov[u] = 1;
            for (u32 v = 1; v < N; ++v) {
                if (cov[v]) continue;
                std::vector<u32> W;
                u32 tv = 0;
                for (int t = 0; t < p; ++t, tv = pa_addidx(p, n, tv, v))
                    for (u32 u : U) W.push_back(pa_addidx(p, n, u, tv));
                std::sort(W.begin(), W.end());
                for (u32 w : W) cov[w] = 1;
                next.insert(W);
            }
        }
        L[k + 1].assign(next.begin(), next.end());
    }
    return L;
}

// Largest k such that f is constant on a k-flat, trying every flat.
static int pa_brute(const PT& f, int p, int n) {
    const auto& L = pa_all_subspaces(p, n);
    const u32 N = (u32)f.size();
    for (int k = n; k >= 0; --k)
        for (const auto& U : L[k])
            for (u32 a = 0; a < N; ++a) {
                bool ok = true;
                for (size_t i = 1; i < U.size() && ok; ++i) ok = f[pa_addidx(p, n, a, U[i])] == f[a];
                if (ok) return k;
            }
    return -1;
}

static int pary_selftest(int& checks) {
    int fails = 0;
    double tlap = now_sec();
    auto lap = [&](const char* what) {
        printf("  %-46s %6.1f s\n", what, now_sec() - tlap);
        fflush(stdout);
        tlap = now_sec();
    };
    auto expect = [&](bool cond, const char* what, int p, int n, u64 seed) {
        ++checks;
        if (!cond) { ++fails; printf("FAIL: %s (p=%d n=%d seed=%llu)\n", what, p, n, (unsigned long long)seed); }
    };
    Options def;
    def.threads = 1;
    Options par = def;
    par.threads = 3;
    std::mt19937_64 rng(99);

    // 1. packed arithmetic, tables, dense quotient, list extraction, cell change
    for (int p : {2, 3, 5, 7, 11, 13, 31, 61}) {
        PField F;
        expect(F.init(p), "field init", p, 0, 0);
        int m = std::min(F.maxdim, 6);
        while (F.pw[m] >= (1ULL << 31)) --m;
        const u64 N = F.pw[m];
        bool ok = true;
        u64 px = 0;
        for (int it = 0; it < 3000; ++it) {
            const u64 a = rng() % N, b = rng() % N;
            const u64 pa = F.pack(a, m), pb = F.pack(b, m);
            ok &= F.unpack(pa, m) == a;
            u64 s = 0, d = 0, w = 1;
            for (int i = 0; i < m; ++i, w *= p) {
                const int x = (int)(a / w % p), y = (int)(b / w % p);
                s += (u64)((x + y) % p) * w;
                d += (u64)((x - y + p) % p) * w;
            }
            ok &= F.unpack(F.add(pa, pb), m) == s && F.unpack(F.sub(pa, pb), m) == d;
            ok &= F.add(pa, pb) == F.pack(s, m);
            const int sc = (int)(rng() % p);
            u64 e = 0;
            for (int t = 0; t < sc; ++t) e = pa_addidx(p, m, (u32)e, (u32)a);
            ok &= F.unpack(F.scale(pa, sc), m) == e;
            if (a) {
                int ld = 0;
                for (u64 t = a; t >= (u64)p; t /= p) ++ld;
                ok &= F.lead(pa) == ld;
            }
            ok &= F.unpack(px, m) == (u64)it % N;
            px = (u64)(it + 1) % N == 0 ? 0 : F.next(px);
        }
        expect(ok, "packed arithmetic", p, m, 0);
    }
    for (int p : {2, 3, 5, 7, 11}) {
        PField F;
        F.init(p);
        std::atomic<bool> stop(false);
        for (int it = 0; it < 60; ++it) {
            const int m = 2 + (int)(rng() % (p == 2 ? 8 : p == 3 ? 6 : p <= 7 ? 4 : 3));
            const int j = 2 + (int)(rng() % (m - 1)), c = F.cdim(j), mc = m - c;  // 2 <= j <= m
            const size_t ncell = F.pw[mc];
            std::vector<u64> T(ncell);
            const u64 cellmask = F.pw[c] == 64 ? ~0ULL : (1ULL << F.pw[c]) - 1;
            for (u64& w : T) w = (rng() | rng() | (it % 3 ? rng() : 0)) & cellmask;
            auto get = [&](const std::vector<u64>& S, int cc, int mm, u64 x) {
                const u64 nc = F.pw[mm - cc];
                return (S[x % nc] >> (x / nc)) & 1;
            };
            PCtx ctx;
            ctx.init(&F, m, &def, &stop, 0);
            // quotient by v = vlow + e_pi
            const int pi = (int)(rng() % (m - j + 1));
            const u32 vl = (u32)(rng() % F.pw[pi]), v = vl + (u32)F.pw[pi];
            std::vector<u32> tr((size_t)(p - 1) * F.pw[pi]);
            ctx.build_tr(tr.data(), pi, vl);
            std::vector<u64> out(F.pw[mc - 1]);
            const u64 blocks = pa_quot(p, T.data(), out.data(), F.pw[pi], F.pw[mc - 1 - pi], tr.data(), 0, F.pw[c]);
            bool ok = true;
            std::set<u64> blk;
            for (u64 y = 0; y < F.pw[m - 1]; ++y) {
                const u64 low = y % F.pw[pi], high = y / F.pw[pi];
                u64 x = low + high * F.pw[pi + 1];
                bool in = true;
                for (int t = 0; t < p; ++t, x = pa_addidx(p, m, (u32)x, v)) in &= get(T, c, m, x) != 0;
                ok &= in == (get(out, c, m - 1, y) != 0);
                if (in) blk.insert(high);
            }
            expect(ok, "dense quotient", p, m, it);
            expect(blocks == blk.size(), "block count of the quotient", p, m, it);
            // list of points
            std::vector<u64> L, ref;
            ctx.extract(T.data(), m, c, L);
            for (u64 x = 0; x < F.pw[m]; ++x)
                if (get(T, c, m, x)) ref.push_back(F.pack(x, m));
            expect(L == ref, "list of a dense set", p, m, it);
            if (c >= 1) {
                std::vector<u64> R;
                ctx.relayout(T.data(), m, c, R);
                bool same = true;
                for (u64 x = 0; x < F.pw[m]; ++x) same &= get(T, c, m, x) == get(R, c - 1, m, x);
                expect(same, "cells of one coordinate less", p, m, it);
            }
        }
    }
    lap("p-ary: primitives");

    // 2. against the brute force, in every mode
    struct Tiny { int p, n, reps; };
    for (const Tiny& t : {Tiny{2, 5, 40}, Tiny{3, 1, 6}, Tiny{3, 2, 30}, Tiny{3, 3, 60}, Tiny{3, 4, 60}, Tiny{3, 5, 10},
                          Tiny{5, 2, 30}, Tiny{5, 3, 40}, Tiny{5, 4, 6}, Tiny{7, 2, 30}, Tiny{7, 3, 20}, Tiny{11, 2, 20},
                          Tiny{13, 2, 10}}) {
        PField F;
        F.init(t.p);
        for (int it = 0; it < t.reps; ++it) {
            PT f = pa_random(t.p, t.n, 1000 * t.p + 10 * t.n + it);
            if (it % 4 == 1)  // few values: large flats
                for (auto& v : f) v = v % 2;
            if (it % 4 == 2)
                for (auto& v : f) v = v < t.p / 2 + 1 ? 0 : v;
            if (it % 4 == 3 && t.n >= 2) f = pa_plant(f, t.p, t.n, 1 + it % t.n, it % t.p, it);
            const int kb = pa_brute(f, t.p, t.n);
            for (int mode = 0; mode < 4; ++mode) {
                PResult r = pa_solve(f, t.p, t.n, mode == 1 ? par : def, -1, mode);
                expect(r.k == kb, "solver vs brute force", t.p, t.n, 10 * it + mode);
                expect((int)r.basis.size() == r.k && pa_constant_on_flat(F, f.data(), t.n, r.a, r.basis, r.value),
                       "witness", t.p, t.n, 10 * it + mode);
                bool ubok = true;
                for (int c = 0; c < t.p; ++c) ubok &= r.ub[c] <= t.n;
                expect(ubok && r.greedy_k <= kb, "bounds", t.p, t.n, it);
                // the searches themselves, without the bounds
                Stats st;
                PFlat fl;
                bool any = false, above = false;
                for (int c = 0; c < t.p; ++c) {
                    any |= pa_decide(F, f.data(), t.n, c, kb, def, fl, st, mode);
                    if (kb < t.n) above |= pa_decide(F, f.data(), t.n, c, kb + 1, def, fl, st, mode);
                }
                expect(any && !above, "decision search vs brute force", t.p, t.n, 10 * it + mode);
            }
            const int tg = 1 + it % t.n;
            PResult r = pa_solve(f, t.p, t.n, def, tg);
            expect((r.k >= tg) == (kb >= tg), "target", t.p, t.n, it);
        }
    }
    lap("p-ary: against brute force");

    // 3. p = 2 against the binary solver
    {
        struct BC { TT F; int n; };
        std::vector<BC> bc;
        for (int n : {6, 7, 8, 9}) {
            bc.push_back({random_function(n, 5 + n), n});
            bc.push_back({random_function(n, 50 + n, 1, 4), n});
            bc.push_back({random_degree(n, 2, 500 + n), n});
        }
        bc.push_back({kasami(8), 8});
        bc.push_back({kasami(10), 10});
        bc.push_back({add_linear(kasami(10), 10, 1), 10});
        bc.push_back({mm_bent(8, 3), 8});
        bc.push_back({threshold_function(9, 5), 9});
        int idx = 0;
        for (const BC& c : bc) {
            ++idx;
            PT f((size_t)1 << c.n);
            for (u32 x = 0; x < f.size(); ++x) f[x] = getbit(c.F.data(), x);
            const int kb = solve(c.F, c.n, def).k;
            for (int mode = 0; mode < 4; ++mode)
                expect(pa_solve(f, 2, c.n, mode == 2 ? par : def, -1, mode).k == kb, "p = 2 vs the binary solver", 2, c.n,
                       10 * idx + mode);
        }
    }
    lap("p-ary: p = 2 against the binary solver");

    // 4. closed forms
    struct QC { int p, n; };
    for (const QC& q : {QC{3, 4}, QC{3, 5}, QC{3, 6}, QC{3, 7}, QC{3, 8}, QC{5, 4}, QC{5, 5}, QC{5, 6}, QC{7, 4}, QC{7, 5},
                        QC{11, 4}, QC{13, 3}}) {
        PField F;
        F.init(q.p);
        for (int type = 0; type < (q.n % 2 ? 1 : 2); ++type) {
            const PT f = pa_scramble(pa_quadratic(q.p, q.n, type), q.p, q.n, 7 * q.n + type);
            const int want = q.n % 2 ? (q.n - 1) / 2 : q.n / 2 - type;
            for (int mode = 0; mode < 4; ++mode) {
                PResult r = pa_solve(f, q.p, q.n, mode ? def : par, -1, mode);
                expect(r.k == want, "quadratic form", q.p, q.n, 10 * type + mode);
                expect(r.bent, "quadratic form is bent", q.p, q.n, 10 * type + mode);
                Stats st;
                PFlat fl;
                bool above = false;
                for (int c = 0; c < q.p; ++c) above |= pa_decide(F, f.data(), q.n, c, want + 1, par, fl, st, mode);
                expect(!above, "quadratic form: search above the maximum", q.p, q.n, 10 * type + mode);
            }
        }
    }
    for (const QC& q : {QC{3, 4}, QC{3, 6}, QC{3, 8}, QC{5, 4}, QC{5, 6}, QC{7, 4}}) {
        const PT f = pa_scramble(pa_mm(q.p, q.n, q.n), q.p, q.n, 31 + q.n);
        for (int mode = 0; mode < 4; ++mode) {
            PResult r = pa_solve(f, q.p, q.n, def, -1, mode);
            expect(r.k == q.n / 2 && r.bent, "Maiorana-McFarland", q.p, q.n, mode);
        }
    }
    lap("p-ary: quadratic forms, Maiorana-McFarland");

    // 5. the modes against each other on larger cases: planted flats, few values, field functions
    struct MC { int p, n, k; };
    int mi = 0;
    for (const MC& c : {MC{3, 6, 3}, MC{3, 7, 4}, MC{3, 8, 4}, MC{3, 9, 5}, MC{5, 5, 3}, MC{5, 6, 4}, MC{7, 4, 2}, MC{7, 5, 3},
                        MC{11, 3, 2}, MC{13, 4, 2}}) {
        PField F;
        F.init(c.p);
        for (int it = 0; it < 3; ++it) {
            ++mi;
            PT f = pa_random(c.p, c.n, 300 + mi);
            if (it == 1)
                for (auto& v : f) v = v % 2;
            f = pa_plant(f, c.p, c.n, c.k - (it == 2), (int)(rng() % c.p), 17 * mi);
            int k0 = -1;
            for (int mode = 0; mode < 4; ++mode) {
                PResult r = pa_solve(f, c.p, c.n, mode == 0 ? par : def, -1, mode);
                if (mode == 0) k0 = r.k;
                expect(r.k == k0 && r.k >= c.k - (it == 2), "modes agree (planted flat)", c.p, c.n, 10 * mi + mode);
            }
        }
    }
    {
        struct FC { int p, n; u64 d; };
        for (const FC& c : {FC{3, 6, 2}, FC{3, 6, 4}, FC{3, 6, 14}, FC{3, 7, 14}, FC{3, 8, 14}, FC{5, 4, 2}, FC{5, 4, 6},
                            FC{5, 5, 6}, FC{7, 4, 8}}) {
            PT f;
            expect(pa_add_monomial(f, c.p, c.n, c.d, 0), "field construction", c.p, c.n, c.d);
            int k0 = -1;
            for (int mode = 0; mode < 4; ++mode) {
                PResult r = pa_solve(f, c.p, c.n, par, -1, mode);
                if (mode == 0) k0 = r.k;
                expect(r.k == k0, "modes agree (trace monomial)", c.p, c.n, 10 * c.d + mode);
            }
            // the spectral bounds stopped the searches at k0: no larger flat, by exhaustive search
            PField F;
            F.init(c.p);
            Stats st;
            PFlat fl;
            bool above = false;
            for (int cst = 0; cst < c.p; ++cst) above |= pa_decide(F, f.data(), c.n, cst, k0 + 1, par, fl, st);
            expect(!above, "trace monomial: search above the maximum", c.p, c.n, c.d);
        }
        // Tr(x^2) is a non-degenerate quadratic form
        for (const QC& q : {QC{3, 4}, QC{3, 5}, QC{5, 4}}) {
            PT f;
            pa_add_monomial(f, q.p, q.n, 2, 0);
            PResult r = pa_solve(f, q.p, q.n, def);
            expect(r.bent && (q.n % 2 ? r.k == (q.n - 1) / 2 : r.k == q.n / 2 || r.k == q.n / 2 - 1), "Tr(x^2)", q.p, q.n, 0);
        }
    }
    lap("p-ary: modes against each other");

    // 5b. with almost no memory for the lines of a pass, so that the directions are split into classes
    {
        pa_rec_limit = 25;
        int idx = 0;
        for (const MC& c : {MC{3, 6, 3}, MC{3, 7, 3}, MC{5, 4, 2}, MC{7, 3, 2}, MC{2, 8, 4}}) {
            for (int it = 0; it < 4; ++it) {
                ++idx;
                PT f = pa_random(c.p, c.n, 700 + idx);
                if (it % 2)
                    for (auto& v : f) v = v % 2;
                if (it >= 2) f = pa_plant(f, c.p, c.n, c.k, 0, idx);
                pa_rec_limit = (size_t)1 << 22;
                const int k0 = pa_solve(f, c.p, c.n, def).k;
                pa_rec_limit = 25;
                for (int mode = 0; mode < 2; ++mode)
                    expect(pa_solve(f, c.p, c.n, mode ? def : par, -1, mode).k == k0, "directions split into classes", c.p,
                           c.n, 10 * idx + mode);
            }
        }
        pa_rec_limit = (size_t)1 << 22;
    }
    lap("p-ary: little memory for lines");

    // 6. invariant linear structures: g(x) = f(the first coordinates of A x)
    for (const QC& q : {QC{3, 6}, QC{5, 4}, QC{7, 3}}) {
        for (int drop = 1; drop <= 2; ++drop) {
            PField F;
            F.init(q.p);
            const int m = q.n - drop;
            const PT g = pa_random(q.p, m, 40 + q.n + drop);
            PT f(F.pw[q.n]);
            for (size_t x = 0; x < f.size(); ++x) f[x] = g[x % F.pw[m]];
            f = pa_scramble(f, q.p, q.n, 3 + drop);
            PResult r = pa_solve(f, q.p, q.n, def), rg = pa_solve(g, q.p, m, def);
            expect(r.lin_dim == drop && r.k == rg.k + drop, "linear structures", q.p, q.n, drop);
        }
    }
    {
        PT f(27, 2);
        PResult r = pa_solve(f, 3, 3, def);
        expect(r.k == 3 && r.value == 2, "constant function", 3, 3, 0);
    }
    lap("p-ary: linear structures");
    return fails;
}
