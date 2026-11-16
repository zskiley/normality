// Symmetry breaking with the automorphism group of the function.
//
// If the affine map A(x) = Mx + b maps a set S onto itself, it maps flats in S
// to flats in S, and the direction space E of a flat to M E.  The direction
// spaces the search is looking for are therefore permuted by the group G of
// the linear parts M, and it is enough to look at one of them per orbit.
//
// The search identifies E with the sequence (w_1, w_2, ...) of its echelon
// basis, each vector written in the quotient coordinates of its level: w_1 is
// the smallest non-zero element of E (as an integer), w_2 the smallest
// non-zero element of E/<w_1>, and so on.  Take the E of an orbit whose
// sequence is lexicographically least.  Then
//     w_1 <= M w_1    for every M in G,
//     w_2 <= M' w_2   for every M in the stabiliser of w_1 (M' = the map it
//                     induces on the quotient by w_1),
// and so on down the chain of stabilisers.  So at every level a candidate
// direction can be skipped if some element of the current stabiliser maps it
// to a smaller vector.  Testing only some elements of the stabiliser is still
// sound; it merely prunes less.
//
// The first condition says more than that: w_1 <= M x for every M in G and
// every non-zero x in E, because M x lies in M E, whose sequence is not
// smaller.  So no vector of E may lie in an orbit that contains a vector
// below w_1.  With low(w_1) = the union of those orbits, a direction can be
// skipped at any level if its coset modulo the directions chosen so far meets
// low(w_1).  That set of cosets is the image of low(w_1) in the quotient, kept
// as one more bitset per level.  It needs only the orbits, so it works
// equally for groups far too large to list.
//
// The second condition can be used with the whole group as well.  If E holds
// another vector u of the orbit of w_1, and M u = w_1, then M E also starts
// with w_1, so its second entry is not smaller than w_2: for every x in E
// outside {0, u}, the class of M x modulo w_1 is at least w_2, and so is
// everything the stabiliser of w_1 makes of it.  Each such u ("mate" of w_1)
// thus gives one more view of E to compare with, and one more set of vectors
// that E must avoid.  M is read off the spanning tree of the orbit.
//
// Nothing here has to trust where the group came from: every map is checked
// against the truth table before use.
#pragma once
#include "bitops.hpp"
#include <algorithm>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// A linear map on F_2^m by columns: M x = xor of c[i] over the set bits i of x.
struct LMap {
    u32 c[32];
};

static inline u32 lm_apply(const LMap& M, u32 x) {
    u32 r = 0;
    for (; x; x &= x - 1) r ^= M.c[__builtin_ctz(x)];
    return r;
}
static inline LMap lm_identity(int m) {
    LMap I;
    for (int i = 0; i < 32; ++i) I.c[i] = i < m ? 1u << i : 0;
    return I;
}
static inline bool lm_is_identity(const LMap& M, int m) {
    for (int i = 0; i < m; ++i)
        if (M.c[i] != 1u << i) return false;
    return true;
}
static inline bool lm_equal(const LMap& A, const LMap& B, int m) { return !memcmp(A.c, B.c, m * sizeof(u32)); }
static inline LMap lm_mul(const LMap& A, const LMap& B, int m) {  // A after B
    LMap R;
    for (int i = 0; i < 32; ++i) R.c[i] = i < m ? lm_apply(A, B.c[i]) : 0;
    return R;
}
static inline u64 lm_hash(const LMap& M, int m) {
    u64 h = 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < m; ++i) {
        h ^= M.c[i];
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 29;
    }
    return h;
}
static bool lm_inverse(const LMap& M, int m, LMap& out) {
    u32 vec[32], pre[32];  // by leading bit of vec: M(pre) = vec
    bool have[32];
    for (int i = 0; i < m; ++i) have[i] = false;
    for (int i = 0; i < m; ++i) {
        u32 v = M.c[i], q = 1u << i;
        if (m < 32 && (v >> m)) return false;
        while (v) {
            int h = msb32(v);
            if (!have[h]) { have[h] = true; vec[h] = v; pre[h] = q; break; }
            v ^= vec[h];
            q ^= pre[h];
        }
        if (!v) return false;
    }
    for (int h = 0; h < m; ++h) {  // clear the bits below the leading one
        u32 v = vec[h] ^ (1u << h), q = pre[h];
        for (; v; v &= v - 1) q ^= out.c[__builtin_ctz(v)];
        out.c[h] = q;
    }
    for (int i = m; i < 32; ++i) out.c[i] = 0;
    return true;
}
// The map induced on F_2^m / <v> (coordinate p = leading bit of v deleted) by M with M v = v.
static inline LMap lm_induced(const LMap& M, int m, u32 v, int p) {
    LMap R;
    for (int i = 0; i < m - 1; ++i) {
        u32 y = M.c[i < p ? i : i + 1];
        if ((y >> p) & 1) y ^= v;
        R.c[i] = remove_bit(y, p);
    }
    for (int i = m - 1; i < 32; ++i) R.c[i] = 0;
    return R;
}
static void lm_dedupe(std::vector<LMap>& L, int m) {
    if (L.size() < 2) return;
    std::sort(L.begin(), L.end(), [m](const LMap& a, const LMap& b) { return memcmp(a.c, b.c, m * sizeof(u32)) < 0; });
    L.erase(std::unique(L.begin(), L.end(), [m](const LMap& a, const LMap& b) { return lm_equal(a, b, m); }), L.end());
}

// All elements of the group generated by `gens` except the identity, if there
// are at most `cap` of them.
static bool lm_close(const std::vector<LMap>& gens, int m, size_t cap, std::vector<LMap>& out) {
    std::vector<LMap> el;
    std::unordered_set<u64> seen;
    el.push_back(lm_identity(m));
    seen.insert(lm_hash(el[0], m));
    for (size_t i = 0; i < el.size(); ++i)
        for (const LMap& g : gens) {
            LMap e = lm_mul(g, el[i], m);
            if (seen.insert(lm_hash(e, m)).second) {
                if (el.size() > cap) return false;
                el.push_back(e);
            }
        }
    out.assign(el.begin() + 1, el.end());
    return true;
}

// f on m variables as a bitset (bit x = f(x)).  Returns 0 if f(Mx + t) = f(x)
// for all x, 1 if f(Mx + t) = f(x) + 1 for all x, and -1 otherwise.
static int aut_kind(const u64* F, int m, const LMap& M, u32 t) {
    LMap inv;
    if (!lm_inverse(M, m, inv) || (m < 32 && (t >> m))) return -1;
    bool same = true, flip = true;
    u32 img = t;
    for (u32 g = 0;; ++g) {
        u32 x = g ^ (g >> 1);
        bool e = getbit(F, x) == getbit(F, img);
        same &= e;
        flip &= !e;
        if (!same && !flip) return -1;
        if (g + 1 == (1u << m)) break;
        img ^= M.c[__builtin_ctz(g + 1)];
    }
    return same ? 0 : 1;
}

// An affine map x -> Mx + t on F_2^n (col = the columns of M), offered as a
// possible automorphism of a function: f(Mx + t) = f(x) or f(x) + 1 for all x.
struct AffMap {
    std::vector<u32> col;
    u32 t = 0;
};
// A computation, running elsewhere, of generators of the group of all such
// maps of a function.  poll() returns 0 while it is still running, 1 once
// with the generators, and -1 if there will be no answer; `note` then says
// what happened.  stop() ends it early.  Destroying the job stops it too.
struct AutJob {
    virtual ~AutJob() {}
    virtual int poll(std::vector<AffMap>& gens, std::string& note) = 0;
    virtual void stop(std::string& note) = 0;
};
// Starts such a computation for f (bit x of F = f(x), m variables), to take
// at most about `budget` seconds; null if it cannot be started.
typedef AutJob* (*AutStart)(const u64* F, int m, double budget);

// A subgroup of GL(m,2) in one coordinate system.
struct SymGroup {
    int m = 0;
    std::vector<LMap> gens;   // generators, identity excluded
    std::vector<LMap> all;    // every element but the identity, if `full`
    bool full = false;
    std::vector<u64> is_min;  // bit v: v is the smallest point of its orbit
    std::vector<u32> omin;    // omin[x] = the smallest point of the orbit of x
    // Spanning trees of the orbits, rooted at their smallest points: x was
    // reached from ginv[back[x]](x) by the generator gens[back[x]].
    std::vector<unsigned char> back;
    std::vector<LMap> ginv;
    bool can_transport() const { return !back.empty(); }
    bool trivial() const { return gens.empty(); }
    const std::vector<LMap>& list() const { return full ? all : gens; }
    u32 count_roots(u32 vend) const {  // orbit representatives among 1 .. vend-1
        u32 c = 0;
        for (u32 v = 1; v < vend; ++v) c += getbit(is_min.data(), v);
        return c;
    }
};

static const size_t SYM_ROOT_CAP = (size_t)1 << 12;  // list a group up to this order
static const size_t SYM_NODE_CAP = 1024;             // the same for stabilisers found by random elements

// Orbits of the group generated by `gens` on F_2^m: id[x] = orbit number, size[o] = orbit size.
static void sym_orbits(const std::vector<LMap>& gens, int m, std::vector<u32>& id, std::vector<u32>& size) {
    size_t N = (size_t)1 << m;
    id.assign(N, ~0u);
    size.clear();
    std::vector<u32> q;
    for (u32 s = 0; s < N; ++s) {
        if (id[s] != ~0u) continue;
        u32 o = (u32)size.size();
        q.clear();
        q.push_back(s);
        id[s] = o;
        for (size_t i = 0; i < q.size(); ++i) {
            u32 x = q[i];
            for (const LMap& g : gens) {
                u32 y = lm_apply(g, x);
                if (id[y] == ~0u) { id[y] = o; q.push_back(y); }
            }
        }
        size.push_back((u32)q.size());
    }
}

static void sym_finish(SymGroup& G, size_t cap = SYM_ROOT_CAP) {
    int m = G.m;
    lm_dedupe(G.gens, m);
    G.gens.erase(std::remove_if(G.gens.begin(), G.gens.end(), [m](const LMap& g) { return lm_is_identity(g, m); }),
                 G.gens.end());
    G.full = !G.gens.empty() && cap > 0 && lm_close(G.gens, m, cap, G.all);
    if (!G.full) G.all.clear();
    size_t N = (size_t)1 << m;
    G.is_min.assign(nwords(m), 0);
    G.omin.assign(N, ~0u);
    G.ginv.clear();
    for (const LMap& g : G.gens) {
        LMap gi;
        lm_inverse(g, m, gi);
        G.ginv.push_back(gi);
    }
    G.back.clear();
    if (!G.gens.empty() && G.gens.size() <= 250) G.back.assign(N, 0);
    std::vector<u32> q;
    for (u32 s = 0; s < N; ++s) {  // orbits in the order of their smallest point
        if (G.omin[s] != ~0u) continue;
        G.is_min[s >> 6] |= 1ULL << (s & 63);
        G.omin[s] = s;
        q.clear();
        q.push_back(s);
        for (size_t i = 0; i < q.size(); ++i) {
            u32 x = q[i];
            for (size_t k = 0; k < G.gens.size(); ++k) {
                u32 y = lm_apply(G.gens[k], x);
                if (G.omin[y] == ~0u) {
                    G.omin[y] = s;
                    if (!G.back.empty()) G.back[y] = (unsigned char)k;
                    q.push_back(y);
                }
            }
        }
    }
}

// The group element M, a product of inverse generators along the orbit's
// spanning tree, that takes u to the smallest point of its orbit.
static void sym_transport(const SymGroup& G, u32 u, LMap& M) {
    M = lm_identity(G.m);
    const u32 root = G.omin[u];
    for (u32 x = u; x != root;) {
        const LMap& gi = G.ginv[G.back[x]];
        x = lm_apply(gi, x);
        M = lm_mul(gi, M, G.m);
    }
}

// The same group in the coordinates y with x = R y (col = the columns of R).
static SymGroup sym_conjugate(const SymGroup& G, const std::vector<u32>& col) {
    SymGroup H;
    H.m = G.m;
    LMap R, Ri;
    for (int i = 0; i < 32; ++i) R.c[i] = i < G.m ? col[i] : 0;
    if (!lm_inverse(R, G.m, Ri)) return H;
    for (const LMap& g : G.gens) H.gens.push_back(lm_mul(Ri, lm_mul(g, R, G.m), G.m));
    sym_finish(H);
    return H;
}

// A basis b_0, b_1, ... of F_2^m whose low spans meet few orbits.  The search
// takes its first direction from span(b_0..b_(c-1)), one vector per orbit, so
// few orbits there means few root candidates.  Greedy: the next vector is the
// candidate that adds the fewest new orbits; ties go to the smaller orbit,
// fixed vectors last.
static std::vector<u32> sym_adapted_basis(const std::vector<LMap>& gens, int m) {
    size_t N = (size_t)1 << m;
    std::vector<u32> id, size;
    sym_orbits(gens, m, id, size);
    // candidates: a spread of points from every class of orbit sizes, small
    // orbits first, and the vectors (g + 1) e_i, which span the space the group moves
    std::vector<u32> cand;
    {
        std::unordered_map<u32, u32> taken;
        const u32 PER = 192;
        u32 mult = 0x9E3779B1u | 1u, mask = (u32)N - 1;
        for (u32 i = 1; i < N && cand.size() < 6000; ++i) {
            u32 x = (i * mult) & mask;  // odd multiplier: a permutation of the points
            if (!x) continue;
            u32& t = taken[size[id[x]]];
            if (t < PER) { ++t; cand.push_back(x); }
        }
        for (const LMap& g : gens)
            for (int i = 0; i < m; ++i) {
                u32 x = g.c[i] ^ (1u << i);
                if (x) cand.push_back(x);
            }
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
    }
    std::vector<u32> basis, ech(m, 0), P(1, 0), stamp(size.size(), 0);
    std::vector<char> have(size.size(), 0);
    have[id[0]] = 1;
    auto independent = [&](u32 v) {
        while (v) {
            int h = msb32(v);
            if (!ech[h]) return true;
            v ^= ech[h];
        }
        return false;
    };
    auto add = [&](u32 b) {
        basis.push_back(b);
        u32 v = b;
        while (v) {
            int h = msb32(v);
            if (!ech[h]) { ech[h] = v; break; }
            v ^= ech[h];
        }
        size_t n0 = P.size();
        for (size_t i = 0; i < n0; ++i) {
            P.push_back(P[i] ^ b);
            have[id[P[i] ^ b]] = 1;
        }
    };
    u32 tick = 0;
    while ((int)basis.size() < m && P.size() <= ((size_t)1 << 12)) {
        u32 best = 0;
        u64 bestkey = ~0ULL;
        for (u32 b : cand) {
            if (!independent(b)) continue;
            ++tick;
            u32 fresh = 0;
            for (u32 x : P) {
                u32 o = id[x ^ b];
                if (!have[o] && stamp[o] != tick) { stamp[o] = tick; ++fresh; }
            }
            u32 os = size[id[b]];
            u64 key = ((u64)fresh << 32) | (os == 1 ? 0xFFFFFFFFu : os);
            if (key < bestkey) { bestkey = key; best = b; }
        }
        if (!best) break;
        add(best);
    }
    // The rest of the basis is free, and so is the choice of basis inside the
    // nested spans.  Both are randomised: structure aligned with the
    // coordinates tends to make the search's other prunings worse.
    std::mt19937_64 rng(20240229);
    u32 lim = m >= 32 ? ~0u : (((u32)1 << m) - 1);
    P.clear();  // no longer needed; `add` below would only grow it
    while ((int)basis.size() < m) {
        u32 b = (u32)rng() & lim, v = b;
        while (v) {
            int h = msb32(v);
            if (!ech[h]) { ech[h] = v; basis.push_back(b); break; }
            v ^= ech[h];
        }
    }
    for (int i = m - 1; i >= 1; --i) {
        u32 r = (u32)rng();
        for (int k = 0; k < i; ++k)
            if ((r >> k) & 1) basis[i] ^= basis[k];
    }
    return basis;
}

// Per-thread scratch for the stabiliser computations.
struct SymWork {
    std::vector<u64> vis;
    std::vector<u32> q, par;
    std::vector<uint16_t> pg;
    std::vector<LMap> tmp, slots;
    std::vector<u32> tab;
    std::mt19937_64 rng{0x1234567ULL};
    void ensure(int m, bool tree) {
        if (vis.size() < nwords(m)) vis.assign(nwords(m), 0);
        if (tree && par.size() < ((size_t)1 << m)) { par.assign((size_t)1 << m, 0); pg.assign((size_t)1 << m, 0); }
    }
};

// kmin[z] = the smallest point of the orbit of z under the group generated by
// (a sample of) G on F_2^m.  With a sample the orbits may come out too fine,
// which only weakens the pruning.
static void sym_orbit_minima(const std::vector<LMap>& G, int m, std::vector<u32>& kmin, SymWork& w) {
    size_t N = (size_t)1 << m;
    kmin.resize(N);
    const size_t MAXG = 16;
    std::vector<const LMap*> use;
    if (G.size() <= MAXG) for (const LMap& g : G) use.push_back(&g);
    else for (size_t k = 0; k < MAXG; ++k) use.push_back(&G[w.rng() % G.size()]);
    // byte tables: three lookups per image
    int nb = (m + 7) / 8;
    std::vector<u32>& tab = w.tab;
    tab.assign(use.size() * nb * 256, 0);
    for (size_t k = 0; k < use.size(); ++k)
        for (int b = 0; b < nb; ++b) {
            u32* t = tab.data() + (k * nb + b) * 256;
            for (int x = 1; x < 256; ++x) {
                int low = __builtin_ctz(x), col = 8 * b + low;
                t[x] = t[x & (x - 1)] ^ (col < m ? use[k]->c[col] : 0);
            }
        }
    const u32 NONE = ~0u;
    std::fill(kmin.begin(), kmin.end(), NONE);
    std::vector<u32>& q = w.q;
    for (u32 s = 0; s < N; ++s) {
        if (kmin[s] != NONE) continue;
        q.clear();
        q.push_back(s);
        kmin[s] = s;
        for (size_t i = 0; i < q.size(); ++i) {
            u32 x = q[i];
            for (size_t k = 0; k < use.size(); ++k) {
                const u32* t = tab.data() + k * nb * 256;
                u32 y = 0;
                for (int b = 0; b < nb; ++b) y ^= t[b * 256 + ((x >> (8 * b)) & 255)];
                if (kmin[y] == NONE) { kmin[y] = s; q.push_back(y); }
            }
        }
    }
}

// Mark in `skip` (bits 0 .. vend-1) the vectors >= 2^lo that some element of
// the group maps to a smaller vector.  G is either the whole group (`full`) or
// a set of generators; with generators, `exact` computes the true orbit minima
// by breadth-first search, otherwise only the generators themselves are tried.
static void sym_skip(const std::vector<LMap>& G, bool full, bool exact, int m, int lo, u32 vend,
                     std::vector<u64>& skip, SymWork& w) {
    skip.assign(((size_t)vend + 63) >> 6, 0);
    if (full || !exact) {
        if (G.size() <= 32) {  // walk all vectors in Gray order, one xor per image
            for (const LMap& g : G) {
                u32 img = 0;
                for (u32 i = 1; i < vend; ++i) {
                    img ^= g.c[__builtin_ctz(i)];
                    u32 x = i ^ (i >> 1);
                    if (img < x) skip[x >> 6] |= 1ULL << (x & 63);
                }
            }
        } else {               // most vectors are moved down by one of the first few elements
            for (u32 x = (u32)1 << lo; x < vend; ++x)
                for (const LMap& g : G)
                    if (lm_apply(g, x) < x) { skip[x >> 6] |= 1ULL << (x & 63); break; }
        }
        return;
    }
    w.ensure(m, false);
    std::fill(w.vis.begin(), w.vis.begin() + nwords(m), 0);
    for (u32 v = (u32)1 << lo; v < vend; ++v) {
        if ((w.vis[v >> 6] >> (v & 63)) & 1) { skip[v >> 6] |= 1ULL << (v & 63); continue; }
        bool smaller = false;
        w.q.clear();
        w.q.push_back(v);
        w.vis[v >> 6] |= 1ULL << (v & 63);
        for (size_t i = 0; i < w.q.size(); ++i) {
            u32 x = w.q[i];
            for (const LMap& g : G) {
                u32 y = lm_apply(g, x);
                u64& word = w.vis[y >> 6];
                if (!((word >> (y & 63)) & 1)) {
                    word |= 1ULL << (y & 63);
                    w.q.push_back(y);
                    smaller |= y < v;
                }
            }
        }
        if (smaller) skip[v >> 6] |= 1ULL << (v & 63);
    }
    std::fill(w.vis.begin(), w.vis.begin() + nwords(m), 0);  // sym_child expects it clean
}

// Stabiliser of v (leading bit p) in the group given by G on F_2^m, as maps on
// the quotient by v, identity excluded.  G is the whole group (`full`) or a
// set of generators; the return value says which of the two `out` is.
// With generators the stabiliser is generated from random elements h of the
// group: if u takes v to h v along the orbit's spanning tree, u^-1 h fixes v.
// Such a set may generate a proper subgroup, which costs pruning, not soundness.
static bool sym_child(const std::vector<LMap>& G, bool full, int m, u32 v, int p, std::vector<LMap>& out, SymWork& w) {
    out.clear();
    auto take = [&](const LMap& g) {
        LMap h = lm_induced(g, m, v, p);
        if (!lm_is_identity(h, m - 1)) out.push_back(h);
    };
    if (full) {
        for (const LMap& g : G)
            if (lm_apply(g, v) == v) take(g);
        lm_dedupe(out, m - 1);
        return true;
    }
    w.ensure(m, true);
    w.q.clear();
    w.q.push_back(v);
    w.vis[v >> 6] |= 1ULL << (v & 63);
    for (size_t i = 0; i < w.q.size(); ++i) {
        u32 x = w.q[i];
        for (size_t k = 0; k < G.size(); ++k) {
            u32 y = lm_apply(G[k], x);
            u64& word = w.vis[y >> 6];
            if (!((word >> (y & 63)) & 1)) {
                word |= 1ULL << (y & 63);
                w.par[y] = x;
                w.pg[y] = (uint16_t)k;
                w.q.push_back(y);
            }
        }
    }
    for (const LMap& g : G)
        if (lm_apply(g, v) == v) take(g);
    if (w.q.size() > 1) {
        // random elements by product replacement
        std::vector<LMap>& s = w.slots;
        size_t ns = std::max<size_t>(G.size(), 8);
        s.resize(ns);
        for (size_t i = 0; i < ns; ++i) s[i] = G[i % G.size()];
        LMap acc = lm_identity(m);
        auto next = [&]() {
            size_t i = w.rng() % ns, k = w.rng() % (ns - 1);
            if (k >= i) ++k;
            s[i] = (w.rng() & 1) ? lm_mul(s[i], s[k], m) : lm_mul(s[k], s[i], m);
            acc = lm_mul(acc, s[i], m);
        };
        for (int i = 0; i < 40; ++i) next();
        for (int r = 0; r < 20; ++r) {
            next();
            u32 x = lm_apply(acc, v);
            LMap u = lm_identity(m), ui;
            for (u32 y = x; y != v; y = w.par[y]) u = lm_mul(u, G[w.pg[y]], m);
            if (!lm_inverse(u, m, ui)) continue;
            LMap t = lm_mul(ui, acc, m);
            if (lm_apply(t, v) == v) take(t);
        }
    }
    for (u32 x : w.q) w.vis[x >> 6] &= ~(1ULL << (x & 63));
    lm_dedupe(out, m - 1);
    if (out.empty()) return true;
    // List the stabiliser if it is small.  A point with a long orbit shows at once that it is not.
    {
        u32 x0 = 0x2B5A4D93u & (((u32)1 << (m - 1)) - 1);
        if (!x0) x0 = 1;
        w.q.clear();
        w.q.push_back(x0);
        w.vis[x0 >> 6] |= 1ULL << (x0 & 63);
        for (size_t i = 0; i < w.q.size() && w.q.size() <= SYM_NODE_CAP; ++i)
            for (const LMap& g : out) {
                u32 y = lm_apply(g, w.q[i]);
                u64& word = w.vis[y >> 6];
                if (!((word >> (y & 63)) & 1)) { word |= 1ULL << (y & 63); w.q.push_back(y); }
            }
        bool big = w.q.size() > SYM_NODE_CAP;
        for (u32 x : w.q) w.vis[x >> 6] &= ~(1ULL << (x & 63));
        if (big) return false;
    }
    if (lm_close(out, m - 1, SYM_NODE_CAP, w.tmp)) {
        out.swap(w.tmp);
        return true;
    }
    return false;
}
