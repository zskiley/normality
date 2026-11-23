// Self-test of the solver for F_p^n -> F_p: primitives against their
// definitions, the solver against a brute force over all flats (tiny cases),
// against closed-form answers (quadratic forms, Maiorana-McFarland), against
// the binary solver for p = 2, against itself in its search modes, and with
// symmetries against the search without them.  Weak normality is compared
// with the normality of f + <u, x> over all u.
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

    // 7. symmetry: linear algebra
    for (int p : {3, 5, 7}) {
        PField F;
        F.init(p);
        for (int it = 0; it < 30; ++it) {
            const int n = 2 + (int)(rng() % 7);
            // a random matrix, its inverse, minimal polynomials, kernels
            PMat M(n), Mi;
            for (int& v : M.a) v = (int)(rng() % p);
            const bool invertible = pm_inv(M, F, Mi);
            if (invertible) {
                const PMat I = pm_mul(M, Mi, p), J = pm_id(n);
                expect(I.a == J.a, "matrix inverse", p, n, it);
            }
            PVec x(n);
            for (int& v : x) v = (int)(rng() % p);
            x[0] = 1;
            const PPoly q = pp_minpoly(M, x, F);
            const PMat Q = pp_eval(q, M, p);
            bool zero = true;
            for (int v : pm_apply(Q, x, p)) zero &= v == 0;
            expect(zero && q.back() == 1, "minimal polynomial of a vector", p, n, it);
            PPoly prod(1, 1);
            for (const auto& fc : pp_factor(q, p))
                for (int e = 0; e < fc.second; ++e) {
                    PPoly t(prod.size() + fc.first.size() - 1, 0);
                    for (size_t a = 0; a < prod.size(); ++a)
                        for (size_t b = 0; b < fc.first.size(); ++b) t[a + b] = (t[a + b] + prod[a] * fc.first[b]) % p;
                    prod = t;
                }
            expect(prod == q, "factorisation of a polynomial", p, n, it);
            bool kerok = true;
            for (const PVec& v : pm_kernel(Q, F))
                for (int w : pm_apply(Q, v, p)) kerok &= w == 0;
            expect(kerok, "kernel", p, n, it);
            // the rotation of the coordinates (and their reversal): invariant subspaces
            std::vector<PMat> gens(1 + it % 2, PMat(n));
            for (int j = 0; j < n; ++j) gens[0].at((j + 1) % n, j) = 1;
            if (gens.size() > 1)
                for (int j = 0; j < n; ++j) gens[1].at(n - 1 - j, j) = 1;
            const int want = 1 + (int)(rng() % n);
            const PSub W = pa_invariant_subspace(F, n, gens, want, rng);
            bool inv = W.dim() <= want;
            for (const PVec& v : W.b)
                for (const PMat& g : gens) {
                    PSub T = W;
                    inv &= !T.add(pm_apply(g, v, p), F);
                }
            expect(inv, "invariant subspace", p, n, it);
            // the all-ones vector spans one, so dimension 1 is always there; n is the whole space
            if (want == 1) expect(W.dim() == 1, "invariant line of the rotation", p, n, it);
        }
    }
    lap("p-ary: symmetry, linear algebra");

    // 8. symmetry: the solver with maps against the solver without
    auto same_with_maps = [&](const PT& f, int p, int n, const std::vector<PMap>& maps, int want_used, const char* what,
                              u64 seed) {
        const PResult plain = pa_solve(f, p, n, par);
        for (int mode = 0; mode < 4; ++mode) {
            const PResult r = pa_solve(f, p, n, mode % 2 ? def : par, -1, mode, 3, &maps);
            expect(r.k == plain.k, what, p, n, 10 * seed + mode);
            if (r.sym_looked && want_used >= 0) expect(r.sym_used == want_used, "number of maps accepted", p, n, seed);
        }
        if (plain.k < n) {  // nothing above the maximum, with symmetry
            const PResult r = pa_solve(f, p, n, par, plain.k + 1, 0, 3, &maps);
            expect(r.k <= plain.k, "target above the maximum, with symmetry", p, n, seed);
        }
        return plain.k;
    };
    {
        // functions invariant under the rotation (and reversal) of the coordinates, with planted flats
        struct RC { int p, n, k, brute; };
        int idx = 0;
        for (const RC& c : {RC{3, 4, 2, 1}, RC{3, 5, 2, 1}, RC{5, 3, 1, 1}, RC{5, 4, 2, 1}, RC{7, 3, 1, 1}, RC{3, 6, 3, 0},
                            RC{3, 7, 3, 0}, RC{3, 8, 4, 0}, RC{5, 5, 2, 0}, RC{5, 6, 3, 0}, RC{7, 4, 2, 0}, RC{11, 3, 1, 0}}) {
            PField F;
            F.init(c.p);
            const u32 N = (u32)F.pw[c.n];
            for (int it = 0; it < (c.brute ? 6 : 3); ++it) {
                ++idx;
                const bool dihedral = it % 2;
                auto rot = [&](u32 x) { return (u32)(x / c.p + (u64)(x % c.p) * (N / c.p)); };  // digit i -> digit i - 1
                auto rev = [&](u32 x) {
                    u32 y = 0;
                    for (int i = 0; i < c.n; ++i, x /= c.p) y = y * c.p + x % c.p;
                    return y;
                };
                auto orbit = [&](u32 x) {
                    std::vector<u32> o;
                    for (int i = 0; i < c.n; ++i, x = rot(x)) {
                        o.push_back(x);
                        if (dihedral) o.push_back(rev(x));
                    }
                    return o;
                };
                PT f(N);
                for (u32 x = 0; x < N; ++x) {
                    u32 mn = x;
                    for (u32 y : orbit(x)) mn = std::min(mn, y);
                    u64 h = (mn + 1) * 0x9E3779B97F4A7C15ULL + 1000003ULL * idx;
                    h ^= h >> 29;
                    f[x] = (uint8_t)((h >> 7) % (it % 3 == 2 ? 2 : c.p));
                }
                // one value on a random flat and on all its images
                const PT mark = pa_plant(PT(N, 0), c.p, c.n, c.k, 1, 900 + idx);
                const int cv = (int)(rng() % c.p);
                for (u32 x = 0; x < N; ++x)
                    if (mark[x])
                        for (u32 y : orbit(x)) f[y] = (uint8_t)cv;
                std::vector<PMap> maps(dihedral ? 3 : 2);
                for (int j = 0; j < c.n; ++j) {
                    maps[0].col.push_back((u32)F.pw[(j + c.n - 1) % c.n]);       // the rotation
                    maps[1].col.push_back((u32)F.pw[(j + 1) % c.n] + (j == 0));  // not a symmetry (as a rule)
                    if (dihedral) maps[2].col.push_back((u32)F.pw[c.n - 1 - j]);
                }
                const int k0 = same_with_maps(f, c.p, c.n, maps, -1, "rotation-invariant function", idx);
                expect(k0 >= c.k, "planted flat (rotation-invariant)", c.p, c.n, idx);
                if (c.brute) expect(k0 == pa_brute(f, c.p, c.n), "rotation-invariant function vs brute force", c.p, c.n, idx);
            }
        }
    }
    {
        // trace functions with the maps of the field, also in random coordinates
        struct TC { int p, n; u64 d, e, d2; };
        int idx = 0;
        for (const TC& c : {TC{3, 6, 2, 0, 0}, TC{3, 6, 4, 1, 0}, TC{3, 6, 14, 0, 0}, TC{3, 7, 14, 0, 0}, TC{3, 8, 14, 0, 0},
                            TC{3, 8, 14, 1, 0}, TC{3, 8, 802, 0, 2}, TC{3, 9, 4, 0, 0}, TC{5, 4, 2, 0, 0}, TC{5, 4, 6, 1, 0},
                            TC{5, 5, 6, 0, 0}, TC{5, 6, 26, 0, 0}, TC{7, 4, 8, 0, 0}, TC{7, 4, 2, 3, 0}, TC{11, 3, 2, 0, 0},
                            TC{13, 2, 2, 0, 0}}) {
            ++idx;
            PT f;
            pa_add_monomial(f, c.p, c.n, c.d, c.e);
            if (c.d2) pa_add_monomial(f, c.p, c.n, c.d2, 0);
            PGF K;
            K.init(c.p, c.n);
            std::vector<PMap> maps;
            pa_field_maps(K, f, maps);
            PField F;
            F.init(c.p);
            bool all = true;
            for (const PMap& a : maps) {
                PMat M;
                std::vector<int> sg;
                all &= pa_map_check(F, f.data(), c.n, a, M, sg);
            }
            expect(all, "field maps are symmetries", c.p, c.n, idx);
            if (c.e == 0 && c.d2 == 0) expect(!maps.empty(), "a trace monomial has the Frobenius map", c.p, c.n, idx);
            same_with_maps(f, c.p, c.n, maps, (int)maps.size(), "trace function with field maps", idx);
            const PT g = pa_scramble(f, c.p, c.n, 50 + idx);
            pa_scramble_maps(c.p, c.n, 50 + idx, maps);
            same_with_maps(g, c.p, c.n, maps, (int)maps.size(), "trace function with field maps, scrambled", idx);
        }
    }
    {
        // maps that permute the constants
        int looked = 0;
        for (const QC& q : {QC{3, 5}, QC{5, 4}, QC{7, 3}}) {
            PField F;
            F.init(q.p);
            // f(x) = x_0 + g(x_1, ...): the translation by e_0 adds 1
            const PT g = pa_random(q.p, q.n - 1, 90 + q.n);
            PT f(F.pw[q.n]);
            for (size_t x = 0; x < f.size(); ++x) f[x] = (uint8_t)((x % q.p + g[x / q.p]) % q.p);
            std::vector<PMap> maps(1);
            for (int j = 0; j < q.n; ++j) maps[0].col.push_back((u32)F.pw[j]);
            maps[0].t = 1;
            same_with_maps(f, q.p, q.n, maps, 1, "translation that shifts the values", q.n);
            const PResult r = pa_solve(f, q.p, q.n, def, -1, 0, 3, &maps);
            looked += r.sym_looked;
            if (r.sym_looked) expect(r.sym_classes == 1, "all constants in one class", q.p, q.n, 0);
        }
        struct HC { int p, n; u64 d; };
        for (const HC& c : {HC{3, 5, 5}, HC{5, 4, 14}, HC{7, 3, 10}, HC{11, 3, 14}}) {
            PField F;
            F.init(c.p);
            // Tr(x^d): the scalar 2 multiplies it by 2^d
            PT f;
            pa_add_monomial(f, c.p, c.n, c.d, 0);
            std::vector<PMap> maps(1);
            for (int j = 0; j < c.n; ++j) maps[0].col.push_back((u32)(2 * F.pw[j]));
            same_with_maps(f, c.p, c.n, maps, 1, "scalar that multiplies the values", c.d);
            const PResult r = pa_solve(f, c.p, c.n, def, -1, 0, 3, &maps);
            int mu = 1;
            for (u64 i = 0; i < c.d; ++i) mu = mu * 2 % c.p;
            int cls = 0;  // {0} and the cosets of the subgroup generated by 2^d, as far as f takes the values
            std::vector<char> seen(c.p, 0);
            for (int v = 0; v < c.p; ++v) {
                if (seen[v]) continue;
                bool taken = false;
                for (int x = v; !seen[x]; x = x * mu % c.p) seen[x] = 1, taken |= r.ub[x] >= 0;
                cls += taken;
            }
            looked += r.sym_looked;
            if (r.sym_looked) expect(r.sym_classes == cls, "classes of constants under a scalar", c.p, c.n, cls);
        }
        expect(looked >= 3, "the maps were looked at", 0, 0, looked);
    }
    lap("p-ary: symmetry, solver");

    // 9. weak normality (weak.hpp): the largest k such that f is affine on a k-flat is the
    //    maximum over u of the normality of f + <u, x>
    {
        auto witness_ok = [&](const PT& f, int p, int n, const PResult& r) {
            if (!r.weak || (int)r.basis.size() != r.k || r.slope.size() != r.basis.size()) return false;
            u64 total = 1;
            for (int i = 0; i < r.k; ++i) total *= p;
            std::set<u32> seen;
            for (u64 t = 0; t < total; ++t) {
                u32 x = r.a;
                int v = r.value;
                u64 tt = t;
                for (int i = 0; i < r.k; ++i, tt /= p)
                    for (int j = 0; j < (int)(tt % p); ++j) x = pa_addidx(p, n, x, r.basis[i]), v = (v + r.slope[i]) % p;
                if (f[x] != v) return false;
                seen.insert(x);
            }
            return seen.size() == total;  // p^k different points: the basis is independent
        };
        auto plus_linear = [&](const PT& f, int p, int n, u32 u) {
            PT g(f.size());
            for (u32 x = 0; x < g.size(); ++x) {
                int s = f[x];
                u32 a = x, b = u;
                for (int i = 0; i < n; ++i, a /= p, b /= p) s += (int)(a % p) * (int)(b % p);
                g[x] = (uint8_t)(s % p);
            }
            return g;
        };
        struct WC { int p, n, reps, brute; };
        int runs = 0, hits = 0;
        for (const WC& c : {WC{2, 4, 6, 1}, WC{2, 5, 6, 1}, WC{3, 1, 3, 1}, WC{3, 2, 8, 1}, WC{3, 3, 8, 1}, WC{5, 2, 8, 1},
                            WC{7, 2, 6, 1}, WC{11, 2, 4, 1}, WC{2, 7, 3, 0}, WC{3, 4, 6, 0}, WC{3, 5, 3, 0}, WC{5, 3, 5, 0},
                            WC{7, 3, 3, 0}}) {
            for (int it = 0; it < c.reps; ++it) {
                PT f = pa_random(c.p, c.n, 3000 * c.p + 10 * c.n + it);
                if (it % 4 == 1)
                    for (auto& v : f) v = v % 2;
                if (it % 4 == 2)
                    for (auto& v : f) v = v < c.p / 2 + 1 ? 0 : v;
                int planted = 0;
                if (it % 4 == 3 && c.n >= 2) {  // it + x_0 where the last coordinates are 0, in other coordinates
                    planted = 1 + (it / 4) % (c.n - 1);
                    u32 Nk = 1;
                    for (int i = 0; i < planted; ++i) Nk *= c.p;
                    for (u32 x = 0; x < Nk; ++x) f[x] = (uint8_t)((it + x) % c.p);
                    f = pa_scramble(f, c.p, c.n, 40 + it);
                }
                int want = -1;
                for (u32 u = 0; u < f.size(); ++u) {
                    const PT g = plus_linear(f, c.p, c.n, u);
                    want = std::max(want, c.brute ? pa_brute(g, c.p, c.n) : pa_solve(g, c.p, c.n, def).k);
                }
                expect(want >= planted, "weak normality: the planted flat", c.p, c.n, it);
                for (int mode = 0; mode < 4; ++mode) {
                    const PResult r = pa_solve_weak(f, c.p, c.n, mode == 1 ? par : def, -1, mode);
                    expect(r.k == want, "weak normality against all linear functions", c.p, c.n, 10 * it + mode);
                    expect(witness_ok(f, c.p, c.n, r), "weak normality: witness", c.p, c.n, 10 * it + mode);
                    runs += r.graph_dim >= 0;
                    hits += r.graph_found;
                }
                for (int t = 0; t <= c.n; ++t) {
                    const PResult r = pa_solve_weak(f, c.p, c.n, def, t);
                    expect((r.k >= t) == (t <= want) && witness_ok(f, c.p, c.n, r), "weak normality: target", c.p, c.n,
                           100 * it + t);
                }
            }
        }
        expect(hits > 5 && runs - hits > 5, "weak normality: both outcomes of the search in the graph", 0, 0, hits);
        for (int n : {6, 7, 8, 9})
            for (int it = 0; it < 3; ++it) {
                const TT B = it == 2 ? random_degree(n, 3, 40 + n) : random_function(n, 70 + n + it, 1, 2 + 2 * it);
                PT f((size_t)1 << n);
                for (u32 x = 0; x < f.size(); ++x) f[x] = getbit(B.data(), x);
                const PResult r = pa_solve_weak(f, 2, n, def);
                expect(r.k == solve_weak(B, n, def).k && witness_ok(f, 2, n, r), "weak normality: p = 2 vs the binary solver", 2,
                       n, it);
            }
        // a non-degenerate quadratic form is affine only on flats whose directions are singular
        struct WQ { int p, n; };
        for (const WQ& q : {WQ{3, 4}, WQ{3, 5}, WQ{3, 6}, WQ{3, 7}, WQ{5, 4}, WQ{5, 5}, WQ{7, 4}})
            for (int type = 0; type < (q.n % 2 ? 1 : 2); ++type) {
                const PT f = pa_scramble(pa_quadratic(q.p, q.n, type), q.p, q.n, 9 * q.n + type);
                const PResult r = pa_solve_weak(f, q.p, q.n, par);
                expect(r.k == (q.n % 2 ? (q.n - 1) / 2 : q.n / 2 - type) && witness_ok(f, q.p, q.n, r),
                       "weak normality of a quadratic form", q.p, q.n, type);
            }
        for (const WQ& q : {WQ{3, 4}, WQ{3, 6}, WQ{5, 4}}) {
            const PT f = pa_scramble(pa_mm(q.p, q.n, q.n), q.p, q.n, 31 + q.n);
            const PResult r = pa_solve_weak(f, q.p, q.n, def);
            expect(r.k == q.n / 2 && witness_ok(f, q.p, q.n, r), "weak normality of a Maiorana-McFarland function", q.p, q.n, 0);
        }
        // trace functions with the maps of the field: these are symmetries of the graph function too
        struct FC { int p, n; u64 d, e; };
        int fi = 0, in_graph = 0;
        for (const FC& c : {FC{3, 6, 14, 0}, FC{3, 6, 20, 1}, FC{3, 7, 22, 0}, FC{3, 8, 14, 1}, FC{5, 4, 8, 0}, FC{5, 5, 12, 0},
                            FC{5, 4, 26, 1}, FC{7, 3, 10, 0}}) {
            ++fi;
            PField F;
            F.init(c.p);
            PT f;
            PGF K;
            if (!K.init(c.p, c.n) || !pa_add_monomial(f, c.p, c.n, c.d, c.e)) {
                expect(false, "weak normality: no field", c.p, c.n, fi);
                continue;
            }
            std::vector<PMap> maps;
            pa_field_maps(K, f, maps);
            if (fi % 2 == 0) {
                f = pa_scramble(f, c.p, c.n, 5 + fi);
                pa_scramble_maps(c.p, c.n, 5 + fi, maps);
            }
            const PT g = pa_graph_function(F, f, c.n);
            bool all = true;
            for (const PMap& a : pa_graph_maps(F, f, c.n, maps)) {
                PMat M;
                std::vector<int> sg;
                all &= pa_map_check(F, g.data(), c.n + 1, a, M, sg) && sg[0] == 0;
            }
            expect(all, "weak normality: the symmetries carried to the graph", c.p, c.n, fi);
            const PResult plain = pa_solve_weak(f, c.p, c.n, par);
            for (int mode = 0; mode < 4; ++mode) {
                const PResult r = pa_solve_weak(f, c.p, c.n, mode % 2 ? def : par, -1, mode, 3, &maps);
                expect(r.k == plain.k && witness_ok(f, c.p, c.n, r), "weak normality with symmetries", c.p, c.n, 10 * fi + mode);
                in_graph += r.sym_graph && r.sym_used > 0;
            }
        }
        expect(in_graph > 0, "weak normality: no symmetry was used in the graph", 0, 0, 0);
    }
    lap("p-ary: weak normality");
    return fails;
}
