// Symmetry for the solver in characteristic p (included by pary.hpp).
//
// An affine map A(x) = M x + t of F_p^n with f(A x) = sigma(f(x)), sigma a
// permutation of F_p, maps the set {f = c} onto {f = sigma(c)}.  Such maps
// are used in two ways.
//   - Constants in one orbit under the sigmas have isomorphic sets, so only
//     one of them is searched.
//   - The maps with sigma(c) = c permute the flats inside {f = c}.  Among the
//     images of a direction space the search only has to find the one whose
//     first basis vector (its smallest non-zero vector) is smallest, so a
//     first direction is skipped if some map takes it to a smaller one.
// The second rule only bites if the first directions tried, the span of the
// first n - k + 1 coordinates, are mapped to one another.  The coordinates
// are therefore changed so that this span is an invariant subspace of the
// group, if one of that dimension can be found.  Invariant subspaces come
// from the kernels of g(R)^j, g an irreducible factor of the minimal
// polynomial of an element R of the group algebra, closed under the group.
#pragma once
#include <set>

struct PMap {
    std::vector<u32> col;  // the images of e_0 .. e_(n-1), as points
    u32 t = 0;             // the image of 0
};

typedef std::vector<int> PVec;   // the digits of a vector
typedef std::vector<int> PPoly;  // coefficients, lowest first

struct PMat {
    int n = 0;
    std::vector<int> a;  // (M v)_r = sum_c at(r, c) v_c
    PMat() {}
    explicit PMat(int nn) : n(nn), a((size_t)nn * nn, 0) {}
    int& at(int r, int c) { return a[(size_t)r * n + c]; }
    int at(int r, int c) const { return a[(size_t)r * n + c]; }
};

static PMat pm_id(int n) {
    PMat m(n);
    for (int i = 0; i < n; ++i) m.at(i, i) = 1;
    return m;
}
static PMat pm_mul(const PMat& A, const PMat& B, int p) {
    const int n = A.n;
    PMat C(n);
    for (int r = 0; r < n; ++r)
        for (int k = 0; k < n; ++k) {
            const int x = A.at(r, k);
            if (!x) continue;
            for (int c = 0; c < n; ++c) C.at(r, c) = (C.at(r, c) + x * B.at(k, c)) % p;
        }
    return C;
}
static PVec pm_apply(const PMat& A, const PVec& v, int p) {
    PVec w(A.n, 0);
    for (int r = 0; r < A.n; ++r) {
        int s = 0;
        for (int c = 0; c < A.n; ++c) s += A.at(r, c) * v[c];
        w[r] = s % p;
    }
    return w;
}
static bool pm_inv(const PMat& A, const PField& F, PMat& out) {
    const int n = A.n, p = F.p;
    PMat M = A;
    out = pm_id(n);
    for (int c = 0; c < n; ++c) {
        int piv = -1;
        for (int r = c; r < n; ++r)
            if (M.at(r, c)) { piv = r; break; }
        if (piv < 0) return false;
        for (int k = 0; k < n; ++k) {
            std::swap(M.at(piv, k), M.at(c, k));
            std::swap(out.at(piv, k), out.at(c, k));
        }
        const int iv = F.inv[M.at(c, c)];
        for (int k = 0; k < n; ++k) {
            M.at(c, k) = M.at(c, k) * iv % p;
            out.at(c, k) = out.at(c, k) * iv % p;
        }
        for (int r = 0; r < n; ++r) {
            const int x = M.at(r, c);
            if (r == c || !x) continue;
            for (int k = 0; k < n; ++k) {
                M.at(r, k) = (M.at(r, k) + (p - x) * M.at(c, k)) % p;
                out.at(r, k) = (out.at(r, k) + (p - x) * out.at(c, k)) % p;
            }
        }
    }
    return true;
}
static PVec pv_digits(u64 x, int n, int p) {
    PVec v(n);
    for (int i = 0; i < n; ++i, x /= p) v[i] = (int)(x % p);
    return v;
}
static u32 pv_index(const PVec& v, int p) {
    u64 r = 0;
    for (int i = (int)v.size() - 1; i >= 0; --i) r = r * p + v[i];
    return (u32)r;
}
static PMat pm_from_cols(const std::vector<u32>& col, int n, int p) {
    PMat M(n);
    for (int c = 0; c < n; ++c) {
        u32 x = col[c];
        for (int r = 0; r < n; ++r, x /= p) M.at(r, c) = x % p;
    }
    return M;
}
static bool pm_is_scalar(const PMat& M) {
    for (int r = 0; r < M.n; ++r)
        for (int c = 0; c < M.n; ++c)
            if (r != c ? M.at(r, c) != 0 : M.at(r, r) != M.at(0, 0)) return false;
    return true;
}
static std::string pm_key(const PMat& M, const PField& F) {  // the same for scalar multiples
    int iv = 1;
    for (int x : M.a)
        if (x) { iv = F.inv[x]; break; }
    std::string s;
    for (int x : M.a) s.push_back((char)(x * iv % F.p));
    return s;
}

// fn(x, y) for every point x of F_p^n in increasing order, y = M x + t as a
// packed vector; fn returns false to stop.
template <class Fn> static void pa_images(const PField& F, int n, const PMat& M, u32 t, Fn fn) {
    const int p = F.p;
    std::vector<u64> delta(n);  // the change of y when digit i of x goes up and the digits below it wrap around
    u64 acc = 0;
    for (int i = 0; i < n; ++i) {
        u64 c = 0;
        for (int r = 0; r < n; ++r) c |= (u64)M.at(r, i) << (r * F.B);
        delta[i] = F.add(c, acc);
        acc = F.add(acc, c);
    }
    u64 y = F.pack(t, n);
    const u64 N = F.pw[n];
    for (u64 x = 0;;) {
        if (!fn((u32)x, y)) return;
        if (++x == N) return;
        int i = 0;
        for (u64 z = x; z % p == 0; z /= p) ++i;
        y = F.add(y, delta[i]);
    }
}

// Is a an invertible map with f(a(x)) = sigma(f(x)) for all x?  Fills in its linear part and sigma.
static bool pa_map_check(const PField& F, const uint8_t* f, int n, const PMap& a, PMat& M, std::vector<int>& sigma) {
    const int p = F.p;
    const u64 N = F.pw[n];
    if ((int)a.col.size() != n || a.t >= N) return false;
    for (u32 c : a.col)
        if (c >= N) return false;
    M = pm_from_cols(a.col, n, p);
    PMat tmp;
    if (!pm_inv(M, F, tmp)) return false;
    sigma.assign(p, -1);
    bool ok = true;
    pa_images(F, n, M, a.t, [&](u32 x, u64 y) {
        const int u = f[x], w = f[F.unpack(y, n)];
        if (sigma[u] < 0) sigma[u] = w;
        else if (sigma[u] != w) ok = false;
        return ok;
    });
    if (!ok) return false;
    std::vector<char> used(p, 0);
    for (int u = 0; u < p; ++u)
        if (sigma[u] >= 0) {
            if (used[sigma[u]]) return false;
            used[sigma[u]] = 1;
        }
    for (int u = 0, w = 0; u < p; ++u)  // values f does not take: anything that makes sigma a permutation
        if (sigma[u] < 0) {
            while (used[w]) ++w;
            sigma[u] = w;
            used[w] = 1;
        }
    return true;
}

// A subspace of F_p^n by its reduced basis: every vector has a pivot, its
// highest non-zero digit, which is 1 and where the other vectors have 0.
// Sorted by pivot, so equal subspaces have equal bases.
struct PSub {
    std::vector<PVec> b;
    std::vector<int> piv;
    int dim() const { return (int)b.size(); }
    void reduce(PVec& v, int p) const {
        for (size_t i = 0; i < b.size(); ++i) {
            const int c = v[piv[i]];
            if (c)
                for (size_t k = 0; k < v.size(); ++k) v[k] = (v[k] + (p - c) * b[i][k]) % p;
        }
    }
    bool add(PVec v, const PField& F) {  // false if v was in the subspace already
        const int p = F.p, n = (int)v.size();
        reduce(v, p);
        int lead = n - 1;
        while (lead >= 0 && !v[lead]) --lead;
        if (lead < 0) return false;
        const int iv = F.inv[v[lead]];
        for (int& x : v) x = x * iv % p;
        for (PVec& w : b) {
            const int c = w[lead];
            if (c)
                for (int k = 0; k < n; ++k) w[k] = (w[k] + (p - c) * v[k]) % p;
        }
        size_t pos = 0;
        while (pos < piv.size() && piv[pos] < lead) ++pos;
        b.insert(b.begin() + pos, v);
        piv.insert(piv.begin() + pos, lead);
        return true;
    }
    std::string key() const {
        std::string s;
        for (const PVec& v : b)
            for (int x : v) s.push_back((char)x);
        return s;
    }
};

// The smallest subspace containing the seed that every generator maps into itself.
static PSub ps_spin(const std::vector<PVec>& seed, const std::vector<PMat>& gens, const PField& F) {
    PSub U;
    std::vector<PVec> todo;
    for (const PVec& v : seed)
        if (U.add(v, F)) todo.push_back(v);
    for (size_t i = 0; i < todo.size(); ++i)
        for (const PMat& M : gens) {
            PVec w = pm_apply(M, todo[i], F.p);
            if (U.add(w, F)) todo.push_back(w);
        }
    return U;
}

static std::vector<PVec> pm_kernel(const PMat& A, const PField& F) {  // a basis of {v : A v = 0}
    const int n = A.n, p = F.p;
    PMat M = A;
    std::vector<int> where(n, -1);  // the row with its pivot in this column
    int rank = 0;
    for (int c = 0; c < n && rank < n; ++c) {
        int piv = -1;
        for (int r = rank; r < n; ++r)
            if (M.at(r, c)) { piv = r; break; }
        if (piv < 0) continue;
        for (int k = 0; k < n; ++k) std::swap(M.at(piv, k), M.at(rank, k));
        const int iv = F.inv[M.at(rank, c)];
        for (int k = 0; k < n; ++k) M.at(rank, k) = M.at(rank, k) * iv % p;
        for (int r = 0; r < n; ++r) {
            const int x = M.at(r, c);
            if (r == rank || !x) continue;
            for (int k = 0; k < n; ++k) M.at(r, k) = (M.at(r, k) + (p - x) * M.at(rank, k)) % p;
        }
        where[c] = rank++;
    }
    std::vector<PVec> ker;
    for (int c = 0; c < n; ++c) {
        if (where[c] >= 0) continue;
        PVec v(n, 0);
        v[c] = 1;
        for (int c2 = 0; c2 < n; ++c2)
            if (where[c2] >= 0) v[c2] = (p - M.at(where[c2], c)) % p;
        ker.push_back(v);
    }
    return ker;
}

// The monic polynomial q of least degree with q(M) x = 0.
static PPoly pp_minpoly(const PMat& M, PVec x, const PField& F) {
    const int n = M.n, p = F.p;
    std::vector<PVec> rows;
    std::vector<PPoly> pol;  // rows[i] = pol[i](M) x, with a 1 at piv[i]
    std::vector<int> piv;
    for (int d = 0;; ++d) {
        PVec v = x;
        PPoly q(d + 1, 0);
        q[d] = 1;
        for (size_t i = 0; i < rows.size(); ++i) {
            const int c = v[piv[i]];
            if (!c) continue;
            for (int k = 0; k < n; ++k) v[k] = (v[k] + (p - c) * rows[i][k]) % p;
            for (size_t k = 0; k < pol[i].size(); ++k) q[k] = (q[k] + (p - c) * pol[i][k]) % p;
        }
        int lead = -1;
        for (int k = 0; k < n; ++k)
            if (v[k]) { lead = k; break; }
        if (lead < 0) return q;
        const int iv = F.inv[v[lead]];
        for (int& t : v) t = t * iv % p;
        for (int& t : q) t = t * iv % p;
        rows.push_back(v);
        pol.push_back(q);
        piv.push_back(lead);
        x = pm_apply(M, x, p);
    }
}
static bool pp_divide(const PPoly& q, const PPoly& g, int p, PPoly& quot) {  // g monic; true if g divides q
    const int dq = (int)q.size() - 1, dg = (int)g.size() - 1;
    if (dq < dg) return false;
    PPoly r = q;
    quot.assign(dq - dg + 1, 0);
    for (int i = dq; i >= dg; --i) {
        const int c = r[i];
        quot[i - dg] = c;
        if (c)
            for (int k = 0; k <= dg; ++k) r[i - dg + k] = (r[i - dg + k] + (p - c) * g[k]) % p;
    }
    for (int k = 0; k < dg; ++k)
        if (r[k]) return false;
    return true;
}
// Irreducible factors of a monic polynomial with their multiplicities, by trial division.
static std::vector<std::pair<PPoly, int>> pp_factor(PPoly q, int p) {
    std::vector<std::pair<PPoly, int>> res;
    for (int deg = 1; 2 * deg <= (int)q.size() - 1; ++deg) {
        u64 total = 1;
        for (int i = 0; i < deg; ++i) total *= p;
        for (u64 idx = 0; idx < total && 2 * deg <= (int)q.size() - 1; ++idx) {
            PPoly g(deg + 1), quot;
            u64 t = idx;
            for (int i = 0; i < deg; ++i, t /= p) g[i] = (int)(t % p);
            g[deg] = 1;
            int mult = 0;
            while (pp_divide(q, g, p, quot)) {
                q = quot;
                ++mult;
            }
            if (mult) res.push_back({g, mult});
        }
    }
    if (q.size() > 1) {  // what is left has no factor of at most half its degree
        bool same = false;
        for (auto& fc : res)
            if (fc.first == q) ++fc.second, same = true;
        if (!same) res.push_back({q, 1});
    }
    return res;
}
static PMat pp_eval(const PPoly& g, const PMat& M, int p) {
    PMat R(M.n);
    for (int i = (int)g.size() - 1; i >= 0; --i) {
        R = pm_mul(R, M, p);
        for (int d = 0; d < M.n; ++d) R.at(d, d) = (R.at(d, d) + g[i]) % p;
    }
    return R;
}

// A subspace mapped into itself by every generator, of dimension `want` if
// one is found, else the largest one found below that.
static PSub pa_invariant_subspace(const PField& F, int n, const std::vector<PMat>& gens, int want, std::mt19937_64& rng) {
    const int p = F.p;
    std::vector<PSub> pool;
    std::set<std::string> have;
    auto put = [&](const PSub& U) {
        if (U.dim() == 0 || U.dim() >= n || pool.size() >= 200) return;
        if (have.insert(U.key()).second) pool.push_back(U);
    };
    std::vector<PMat> Rs = gens;
    if (gens.size() > 1)
        for (int it = 0; it < 4; ++it) {  // random elements of the group algebra
            PMat R(n);
            for (size_t g = 0; g < gens.size(); ++g) {
                const int c = (int)(rng() % p);
                const PMat& X = it % 2 ? pm_mul(gens[g], gens[rng() % gens.size()], p) : gens[g];
                for (size_t k = 0; k < R.a.size(); ++k) R.a[k] = (R.a[k] + c * X.a[k]) % p;
            }
            Rs.push_back(R);
        }
    for (const PMat& R : Rs) {
        std::vector<std::pair<PPoly, int>> fac;
        for (int it = 0; it < 3; ++it) {
            PVec x(n);
            bool nz = false;
            for (int& v : x) v = (int)(rng() % p), nz |= v != 0;
            if (!nz) x[0] = 1;
            for (const auto& fc : pp_factor(pp_minpoly(R, x, F), p)) {
                bool seen = false;
                for (auto& old : fac)
                    if (old.first == fc.first) old.second = std::max(old.second, fc.second), seen = true;
                if (!seen) fac.push_back(fc);
            }
        }
        for (const auto& fc : fac) {
            const PMat G = pp_eval(fc.first, R, p);
            PMat K = pm_id(n);
            for (int j = 1; j <= fc.second; ++j) {
                K = pm_mul(K, G, p);
                const std::vector<PVec> N = pm_kernel(K, F);
                put(ps_spin(N, gens, F));
                if (j == 1)  // the cyclic pieces inside it
                    for (size_t i = 0; i < N.size() && i < 12; ++i) put(ps_spin({N[i]}, gens, F));
            }
        }
    }
    // sums of pool members, up to the dimension wanted
    std::vector<PSub> states(1);
    std::set<std::string> seen;
    seen.insert(states[0].key());
    PSub best;
    for (size_t i = 0; i < states.size() && states.size() < 3000 && best.dim() < want; ++i)
        for (const PSub& U : pool) {
            PSub T = states[i];
            for (const PVec& v : U.b) T.add(v, F);
            if (T.dim() > want || !seen.insert(T.key()).second) continue;
            if (T.dim() > best.dim()) best = T;
            states.push_back(T);
        }
    return best;
}

// The group generated by gens modulo scalars, without the identity, up to `cap` elements.
static std::vector<PMat> pa_group_elements(const std::vector<PMat>& gens, const PField& F, size_t cap, bool& complete) {
    std::vector<PMat> el;
    if (gens.empty()) { complete = true; return el; }
    std::set<std::string> seen;
    el.push_back(pm_id(gens[0].n));
    seen.insert(pm_key(el[0], F));
    complete = true;
    for (size_t i = 0; i < el.size() && complete; ++i)
        for (const PMat& g : gens) {
            PMat X = pm_mul(g, el[i], F.p);
            if (!seen.insert(pm_key(X, F)).second) continue;
            if (el.size() >= cap + 1) { complete = false; break; }
            el.push_back(X);
        }
    el.erase(el.begin());
    return el;
}

// Bit v set: some element maps the direction v (leading digit 1, leading
// coordinate <= hi) to a smaller one.
static std::vector<u64> pa_root_skip(const PField& F, int hi, const std::vector<PMat>& elems, u64& nskip) {
    const int p = F.p, B = F.B;
    std::vector<u64> bits(F.pw[hi + 1] / 64 + 1, 0);
    const size_t E = elems.size(), row = (size_t)(hi + 1) * p;
    std::vector<u64> mult(E * row, 0);  // the multiples of the columns 0 .. hi of every element, packed
    for (size_t e = 0; e < E; ++e)
        for (int j = 0; j <= hi; ++j) {
            u64 c = 0;
            for (int r = 0; r < elems[e].n; ++r) c |= (u64)elems[e].at(r, j) << (r * B);
            u64* m = mult.data() + e * row + (size_t)j * p;
            for (int t = 1; t < p; ++t) m[t] = F.add(m[t - 1], c);
        }
    nskip = 0;
    for (int pi = 0; pi <= hi; ++pi) {
        u64 pv = 1ULL << (pi * B);
        for (u64 vl = 0; vl < F.pw[pi]; ++vl, pv = F.next(pv)) {
            int dg[48];
            for (int j = 0; j <= pi; ++j) dg[j] = F.digit(pv, j);
            bool skip = false;
            for (size_t e = 0; e < E && !skip; ++e) {
                const u64* m = mult.data() + e * row;
                u64 w = 0;
                for (int j = 0; j <= pi; ++j)
                    if (dg[j]) w = F.add(w, m[(size_t)j * p + dg[j]]);
                const int lw = F.lead(w);
                if (lw < pi) skip = true;
                else if (lw == pi) skip = F.scale(w, F.inv[F.digit(w, pi)]) < pv;
            }
            if (skip) {
                const u64 v = vl + F.pw[pi];
                bits[v >> 6] |= 1ULL << (v & 63);
                ++nskip;
            }
        }
    }
    return bits;
}

// What the maps give for the set {f = c}: the linear parts of maps that take
// the set to itself (generators, and the group they generate modulo scalars
// as far as it was listed).
struct PStab {
    bool ready = false, complete = false;
    std::vector<PMat> gens, elems;
};
static void pa_stabiliser(const PField& F, int c, const std::vector<PMat>& M, const std::vector<std::vector<int>>& sigma,
                          PStab& S) {
    const int p = F.p;
    S.ready = true;
    std::vector<char> have(p, 0);
    std::vector<PMat> T(p);  // T[u]: the linear part of a map taking {f = c} to {f = u}
    std::vector<int> queue(1, c);
    std::set<std::string> seen;
    have[c] = 1;
    T[c] = pm_id(M[0].n);
    for (size_t i = 0; i < queue.size(); ++i) {
        const int u = queue[i];
        for (size_t g = 0; g < M.size(); ++g) {
            const int w = sigma[g][u];
            const PMat X = pm_mul(M[g], T[u], p);
            if (!have[w]) {
                have[w] = 1;
                T[w] = X;
                queue.push_back(w);
                continue;
            }
            PMat Ti;
            pm_inv(T[w], F, Ti);
            const PMat s = pm_mul(Ti, X, p);  // Schreier generator
            if (!pm_is_scalar(s) && seen.insert(pm_key(s, F)).second) S.gens.push_back(s);
        }
    }
    S.elems = pa_group_elements(S.gens, F, 1024, S.complete);
}
