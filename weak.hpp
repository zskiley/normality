// Weak normality: the largest k such that f is affine on some k-dimensional
// flat ("f is weakly k-normal"), for the binary solver and for the solver in
// characteristic p.
//
// The graph of f, {(x, f(x))}, is the set where g(x, y) = f(x) - y is 0; y is
// one more coordinate, the top one.  A flat inside the graph has no two points
// with the same x, so it is the graph of an affine function over its
// projection to the x-space, and conversely.  Hence
//
//     f is affine on a k-flat   <=>   g is constant on a k-flat,
//
// and the solver is run on g.  One constant is enough there: the set g = c is
// the set g = 0 moved along y.
//
// Two bounds are used first, and they leave a single question for the graph.
//   - A function that is affine and not constant on a k-flat is constant on a
//     (k-1)-flat inside it.  So the answer is the normality of f or one more.
//   - If f is affine on a k-flat then p^k <= max |Walsh coefficient of f|.
//     (The general bound of the solver, applied to g, is one higher.)
// So f itself is solved first; then, if the spectrum allows it, it is decided
// whether the graph contains a flat of one more dimension.
//
// Known symmetries of f carry over to g: if f(Ax) = l f(x) + s for all x then
// (x, y) -> (Ax, l y + s) multiplies g by l.
//
// The flat reported is checked point by point against the table of f.
#pragma once
#include "gen.hpp"
#include "pgen.hpp"

// ------------------------------------------------------------------ p = 2

// Is f(a + sum t_i basis[i]) = value + sum t_i slope[i] for all t, and is the basis independent?
static bool affine_on_flat(const u64* F, int n, u32 a, const std::vector<u32>& basis, int value,
                           const std::vector<int>& slope) {
    if (basis.size() != slope.size()) return false;
    std::vector<u32> ech(n, 0);
    for (u32 v : basis) {
        while (v) {
            const int h = msb32(v);
            if (h >= n) return false;
            if (!ech[h]) { ech[h] = v; break; }
            v ^= ech[h];
        }
        if (!v) return false;
    }
    u32 x = a;
    int val = value & 1;
    if ((int)getbit(F, x) != val) return false;
    const u64 total = (u64)1 << basis.size();
    for (u64 g = 1; g < total; ++g) {  // Gray code: one basis vector per step
        const int i = ctz64(g);
        x ^= basis[i];
        val ^= slope[i] & 1;
        if ((int)getbit(F, x) != val) return false;
    }
    return true;
}

// g(x, y) = f(x) + y on n + 1 variables, y = bit n of the point.
static TT graph_function(const TT& F, int n) {
    TT G(nwords(n + 1), 0);
    if (n < 6) {
        const int half = 1 << n;
        const u64 mask = ((u64)1 << half) - 1;
        G[0] = (F[0] & mask) | ((~F[0] & mask) << half);
    } else {
        const size_t w = nwords(n);
        for (size_t i = 0; i < w; ++i) {
            G[i] = F[i];
            G[w + i] = ~F[i];
        }
    }
    return G;
}

// Candidate automorphisms of f as candidates for the graph function.  More
// maps qualify there: if f(Ax) = f(x) + <u, x> + e, then
// (x, y) -> (Ax, y + <u, x> + e) keeps g.  u and e are read off at 0 and at
// the unit vectors; the solver checks the whole map before it uses it.
static std::vector<AffMap> graph_maps(const TT& F, int n, const std::vector<AffMap>& maps) {
    std::vector<AffMap> out;
    const u32 N = (u32)1 << n;
    for (const AffMap& a : maps) {
        bool ok = (int)a.col.size() == n && a.t < N;
        for (u32 c : a.col) ok &= c < N;
        if (!ok) continue;
        const bool e = getbit(F.data(), a.t) != getbit(F.data(), 0);
        AffMap b = a;
        for (int i = 0; i < n; ++i)
            if ((getbit(F.data(), a.col[i] ^ a.t) != getbit(F.data(), (u32)1 << i)) != e) b.col[i] |= N;
        b.col.push_back(N);
        if (e) b.t |= N;
        out.push_back(b);
    }
    return out;
}

// The largest k such that f is affine on a k-flat, as a Result with `weak`
// set: f(a + sum t_i basis[i]) = value + sum t_i slope[i], and at most the
// last slope is not 0.  With target >= 0 only "is f weakly target-normal?" is
// decided: R.k >= target if yes, else the dimension of the best flat found.
// n is at most 29.
static Result solve_weak(const TT& F, int n, const Options& opt, int greedy_restarts = 3, int target = -1) {
    const double t0 = now_sec();
    const int64_t mw = max_walsh(F, n);
    int cap = 0;
    while (cap < n && ((int64_t)2 << cap) <= mw) ++cap;
    // f itself: its normality, or with a target whether it is (target - 1)-normal, which it has to be
    Result R = solve(F, n, opt, greedy_restarts, target < 0 ? -1 : target <= cap ? std::max(target - 1, 0) : 0);
    if (R.k < 0) return R;
    R.weak = true;
    R.walsh_cap = cap;
    R.const_k = R.k;
    R.slope.assign(R.basis.size(), 0);
    const int want = target < 0 ? R.k + 1 : target;
    if (opt.verbose)
        fprintf(stderr, "weak: f is constant on a %d-flat%s; max |Walsh| = %lld allows dimension %d\n", R.k,
                target < 0 ? " and on none larger" : "", (long long)mw, cap);
    if (want <= cap && R.k == want - 1) {
        Options og = opt;
        og.one_const = true;
        std::vector<AffMap> lifted;
        if (opt.sym_cands) lifted = graph_maps(F, n, *opt.sym_cands);
        og.sym_cands = lifted.empty() ? nullptr : &lifted;
        if (opt.verbose) fprintf(stderr, "weak: looking for a %d-flat in the graph of f (%d variables)\n", want, n + 1);
        const Result G = solve(graph_function(F, n), n + 1, og, greedy_restarts, want);
        R.graph_dim = want;
        R.searches += G.searches;
        R.st.add(G.st);
        if (G.sym_loaded) {
            R.sym_graph = true;
            R.sym_loaded = true;
            R.sym_offered = G.sym_offered;
            R.sym_gens = G.sym_gens;
            R.sym_order = G.sym_order;
            R.sym_flip = G.sym_flip;
            R.sym_seconds = G.sym_seconds;
            R.sym_note = G.sym_note;
        }
        if (G.k < 0 || (target < 0 && G.k > want)) {  // more than the normality of f allows: one of the two is wrong
            fprintf(stderr, "INTERNAL ERROR: the search in the graph of f contradicts the normality of f\n");
            R.k = -2;
            return R;
        }
        if (G.k >= want) {
            R.graph_found = true;
            const u32 mask = ((u32)1 << n) - 1;
            R.k = G.k;
            R.a = G.a & mask;
            R.value = G.value ^ (int)(G.a >> n);
            R.basis.clear();
            R.slope.clear();
            for (u32 b : G.basis) {
                R.basis.push_back(b & mask);
                R.slope.push_back((int)(b >> n));
            }
            int last = -1;  // a basis in which only the last vector changes f
            for (int i = 0; i < R.k; ++i)
                if (R.slope[i]) last = i;
            if (last >= 0) {
                for (int i = 0; i < last; ++i)
                    if (R.slope[i]) {
                        R.basis[i] ^= R.basis[last];
                        R.slope[i] = 0;
                    }
                std::rotate(R.basis.begin() + last, R.basis.begin() + last + 1, R.basis.end());
                std::rotate(R.slope.begin() + last, R.slope.begin() + last + 1, R.slope.end());
            }
        }
    }
    R.ub[0] = R.ub[1] = target < 0 ? std::min(cap, R.const_k + 1) : cap;
    if (!affine_on_flat(F.data(), n, R.a, R.basis, R.value, R.slope) || (int)R.basis.size() != R.k) {
        fprintf(stderr, "INTERNAL ERROR: f is not affine on the flat found\n");
        R.k = -2;
    }
    R.seconds = now_sec() - t0;
    return R;
}

// ------------------------------------------------------- characteristic p

static bool pa_affine_on_flat(const PField& F, const uint8_t* f, int n, u32 a, const std::vector<u32>& basis, int value,
                              const std::vector<int>& slope) {
    const int k = (int)basis.size(), p = F.p;
    if ((int)slope.size() != k || a >= F.pw[n]) return false;
    PSub sub;
    for (u32 b : basis)
        if (b >= F.pw[n] || !sub.add(pv_digits(b, n, p), F)) return false;
    std::vector<int> cnt(k, 0);
    u32 x = a;
    int val = ((value % p) + p) % p;
    for (;;) {
        if (f[x] != val) return false;
        int i = 0;
        for (; i < k; ++i) {  // odometer: p times a vector is zero
            x = pa_addidx(p, n, x, basis[i]);
            val = (val + slope[i]) % p;
            if (++cnt[i] < p) break;
            cnt[i] = 0;
        }
        if (i == k) return true;
    }
}

// Can the graph function of a function on n variables be handled?
static bool pa_weak_fits(const PField& F, int n) { return n + 1 <= F.maxdim && F.pw[n + 1] <= 60000000; }

// g(x, y) = f(x) - y on n + 1 variables: the value of the point x + y p^n.
static PT pa_graph_function(const PField& F, const PT& f, int n) {
    const int p = F.p;
    const u32 N = (u32)F.pw[n];
    PT g((size_t)N * p);
    for (int y = 0; y < p; ++y)
        for (u32 x = 0; x < N; ++x) g[(size_t)y * N + x] = (uint8_t)((f[x] + p - y) % p);
    return g;
}

// The candidate maps that are symmetries of f with f(Ax) = l f(x) + s on the
// values f takes, as maps (x, y) -> (Ax, l y + s) of the graph function.
static std::vector<PMap> pa_graph_maps(const PField& F, const PT& f, int n, const std::vector<PMap>& maps) {
    const int p = F.p;
    const u32 N = (u32)F.pw[n];
    std::vector<PMap> out;
    std::vector<char> occurs(p, 0);
    for (u32 x = 0; x < N; ++x) occurs[f[x]] = 1;
    int v1 = -1, v2 = -1;  // the two smallest values of f
    for (int c = p - 1; c >= 0; --c)
        if (occurs[c]) v2 = v1, v1 = c;
    if (v2 < 0) return out;
    for (const PMap& a : maps) {
        PMat M;
        std::vector<int> sg;
        if (!pa_map_check(F, f.data(), n, a, M, sg)) continue;
        const int l = (sg[v2] - sg[v1] + p) % p * F.inv[v2 - v1] % p;
        const int s = ((sg[v1] - l * v1) % p + p) % p;
        bool aff = l != 0;
        for (int c = 0; c < p; ++c)
            if (occurs[c]) aff &= sg[c] == (l * c + s) % p;
        if (!aff) continue;
        PMap b = a;
        b.col.push_back((u32)l * N);
        b.t += (u32)s * N;
        out.push_back(b);
    }
    return out;
}

// As solve_weak, for f : F_p^n -> F_p.  The other arguments are those of pa_solve.
static PResult pa_solve_weak(const PT& f, int p, int n, const Options& opt, int target = -1, int mode = 0, int restarts = 3,
                             const std::vector<PMap>* maps = nullptr) {
    const double t0 = now_sec();
    PField F;
    F.init(p);
    const u32 N = (u32)F.pw[n];
    auto cap_of = [&](double mw) {
        int cap = 0;
        for (double q = p; cap < n && q <= mw * (1 + 1e-9) + 1e-6; q *= p) ++cap;
        return cap;
    };
    int cap = target >= 0 ? cap_of(pa_spectrum(F, f.data(), n, opt.threads).max_walsh) : n;
    PResult R = pa_solve(f, p, n, opt, target < 0 ? -1 : target <= cap ? std::max(target - 1, 0) : 0, mode, restarts, maps);
    if (target < 0) cap = cap_of(R.max_walsh);
    R.weak = true;
    R.walsh_cap = cap;
    R.const_k = R.k;
    R.slope.assign(R.basis.size(), 0);
    const int want = target < 0 ? R.k + 1 : target;
    if (opt.verbose)
        fprintf(stderr, "weak: f is constant on a %d-flat%s; max |Walsh| = %.3f allows dimension %d\n", R.k,
                target < 0 ? " and on none larger" : "", R.max_walsh, cap);
    if (want <= cap && R.k == want - 1) {
        if (!pa_weak_fits(F, n)) {
            fprintf(stderr, "internal error: the graph of f is too large\n");
            abort();
        }
        const PT g = pa_graph_function(F, f, n);
        std::vector<PMap> lifted;
        if (maps) lifted = pa_graph_maps(F, f, n, *maps);
        Options og = opt;
        og.one_const = true;
        if (opt.verbose) fprintf(stderr, "weak: looking for a %d-flat in the graph of f (%d variables)\n", want, n + 1);
        const PResult G = pa_solve(g, p, n + 1, og, want, mode, restarts, lifted.empty() ? nullptr : &lifted);
        R.graph_dim = want;
        R.searches += G.searches;
        R.st.add(G.st);
        if (G.sym_looked) {
            R.sym_graph = true;
            R.sym_looked = true;
            R.sym_offered = G.sym_offered;
            R.sym_used = G.sym_used;
            R.sym_classes = G.sym_classes;
            R.sym_order = G.sym_order;
            R.sym_note = G.sym_note;
        }
        if (target < 0 && G.k > want) {
            fprintf(stderr, "internal error: the search in the graph of f contradicts the normality of f\n");
            abort();
        }
        if (G.k >= want) {
            R.graph_found = true;
            R.k = G.k;
            R.a = G.a % N;
            R.value = (int)((G.value + G.a / N) % p);
            R.basis.clear();
            R.slope.clear();
            for (u32 b : G.basis) {
                R.basis.push_back(b % N);
                R.slope.push_back((int)(b / N));
            }
            auto axpy = [&](u32 x, int c, u32 v) {  // x + c v
                u32 r = 0, w = 1;
                for (int i = 0; i < n; ++i, x /= p, v /= p, w *= p) r += (x % p + c * (v % p)) % p * w;
                return r;
            };
            int last = -1;  // a basis in which only the last vector changes f, by 1
            for (int i = 0; i < R.k; ++i)
                if (R.slope[i]) last = i;
            if (last >= 0) {
                R.basis[last] = axpy(0, F.inv[R.slope[last]], R.basis[last]);
                R.slope[last] = 1;
                for (int i = 0; i < last; ++i)
                    if (R.slope[i]) {
                        R.basis[i] = axpy(R.basis[i], p - R.slope[i], R.basis[last]);
                        R.slope[i] = 0;
                    }
                std::rotate(R.basis.begin() + last, R.basis.begin() + last + 1, R.basis.end());
                std::rotate(R.slope.begin() + last, R.slope.begin() + last + 1, R.slope.end());
            }
        }
    }
    R.ub.assign(p, target < 0 ? std::min(cap, R.const_k + 1) : cap);
    if (!pa_affine_on_flat(F, f.data(), n, R.a, R.basis, R.value, R.slope) || (int)R.basis.size() != R.k) {
        fprintf(stderr, "internal error: f is not affine on the flat found\n");
        abort();
    }
    R.seconds = now_sec() - t0;
    return R;
}
