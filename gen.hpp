// Truth-table input and a few function families for tests and benchmarks.
#pragma once
#include "bitops.hpp"
#include "group.hpp"
#include <random>
#include <string>
#include <unordered_set>

typedef std::vector<u64> TT;  // bit x = f(x), 2^n bits

static inline void setbit(TT& F, u32 x) { F[x >> 6] |= 1ULL << (x & 63); }

// Binary string: character i is f(i).  Returns n, or -1 on bad input.
static int parse_binary(const std::string& s, TT& F) {
    std::string t;
    for (char ch : s)
        if (ch == '0' || ch == '1') t.push_back(ch);
        else if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t' && ch != ',') return -1;
    int n = 0;
    while (((size_t)1 << n) < t.size()) ++n;
    if (t.empty() || ((size_t)1 << n) != t.size() || n > 30) return -1;
    F.assign(nwords(n), 0);
    for (size_t i = 0; i < t.size(); ++i)
        if (t[i] == '1') setbit(F, (u32)i);
    return n;
}

// Hex string read as one big integer whose bit i is f(i) (last digit = f(0..3)).
static int parse_hex(const std::string& s, TT& F) {
    std::string t;
    for (char ch : s) {
        if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t') continue;
        if (!isxdigit((unsigned char)ch)) return -1;
        t.push_back(ch);
    }
    if (t.size() >= 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X')) return -1;
    int n = 2;
    while (((size_t)1 << (n - 2)) < t.size()) ++n;
    if (t.empty() || ((size_t)1 << (n - 2)) != t.size() || n > 30) return -1;
    F.assign(nwords(n), 0);
    for (size_t i = 0; i < t.size(); ++i) {
        char ch = t[t.size() - 1 - i];
        int d = ch <= '9' ? ch - '0' : (ch | 32) - 'a' + 10;
        for (int b = 0; b < 4; ++b)
            if ((d >> b) & 1) setbit(F, (u32)(4 * i + b));
    }
    return n;
}

// Each value is 1 with probability num/den.
static TT random_function(int n, u64 seed, u32 num = 1, u32 den = 2) {
    std::mt19937_64 rng(seed);
    TT F(nwords(n), 0);
    for (u32 x = 0; x < (1u << n); ++x)
        if (rng() % den < num) setbit(F, x);
    return F;
}

// Random function of algebraic degree <= deg (random ANF, Moebius transform).
static TT random_degree(int n, int deg, u64 seed) {
    std::mt19937_64 rng(seed);
    size_t N = (size_t)1 << n;
    std::vector<unsigned char> a(N, 0);
    for (size_t x = 0; x < N; ++x)
        if (__builtin_popcountll(x) <= deg) a[x] = rng() & 1;
    for (int i = 0; i < n; ++i)
        for (size_t x = 0; x < N; ++x)
            if (x >> i & 1) a[x] ^= a[x ^ ((size_t)1 << i)];
    TT F(nwords(n), 0);
    for (size_t x = 0; x < N; ++x)
        if (a[x]) setbit(F, (u32)x);
    return F;
}

// Maiorana-McFarland bent function f(x,y) = <x, pi(y)> + h(y), n = 2t.
// x = low t bits, y = high t bits.  Constant on the t-flat {x = 0... } shifted.
static TT mm_bent(int n, u64 seed) {
    std::mt19937_64 rng(seed);
    int t = n / 2;
    u32 T = 1u << t;
    std::vector<u32> pi(T);
    for (u32 i = 0; i < T; ++i) pi[i] = i;
    for (u32 i = T - 1; i > 0; --i) std::swap(pi[i], pi[rng() % (i + 1)]);
    TT F(nwords(n), 0);
    for (u32 y = 0; y < T; ++y) {
        u32 h = rng() & 1;
        for (u32 x = 0; x < T; ++x)
            if ((__builtin_popcount(x & pi[y]) & 1) ^ h) setbit(F, x | (y << t));
    }
    return F;
}

// Carlet's class D_0: Maiorana-McFarland x.pi(y) plus the indicator of x = 0.
static TT mm_d0(int n, u64 seed) {
    std::mt19937_64 rng(seed);
    int t = n / 2;
    u32 T = 1u << t;
    std::vector<u32> pi(T);
    for (u32 i = 0; i < T; ++i) pi[i] = i;
    for (u32 i = T - 1; i > 0; --i) std::swap(pi[i], pi[rng() % (i + 1)]);
    TT F(nwords(n), 0);
    for (u32 y = 0; y < T; ++y)
        for (u32 x = 0; x < T; ++x)
            if ((__builtin_popcount(x & pi[y]) & 1) ^ (x == 0)) setbit(F, x | (y << t));
    return F;
}

// A random invertible affine map x -> Mx + shift (col = the columns of M).
static void scramble_map(int n, u64 seed, std::vector<u32>& col, u32& shift) {
    std::mt19937_64 rng(seed);
    col.assign(n, 0);
    for (;;) {  // random invertible matrix: columns col[i]
        for (int i = 0; i < n; ++i) col[i] = (u32)(rng() & ((1ull << n) - 1));
        std::vector<u32> b(n, 0);
        int rank = 0;
        for (int i = 0; i < n; ++i) {
            u32 v = col[i];
            while (v) {
                int h = msb32(v);
                if (!b[h]) { b[h] = v; ++rank; break; }
                v ^= b[h];
            }
        }
        if (rank == n) break;
    }
    shift = (u32)(rng() & ((1ull << n) - 1));
}

// Apply a random invertible affine change of variables (hides structure):
// G(x) = F(Mx + shift) with the map of scramble_map.
static TT scramble(const TT& F, int n, u64 seed) {
    std::vector<u32> col;
    u32 shift;
    scramble_map(n, seed, col, shift);
    TT G(nwords(n), 0);
    u32 img = shift;  // image of x under x -> Mx + shift, walked in Gray order
    for (u32 g = 0;; ++g) {
        u32 x = g ^ (g >> 1);
        if (getbit(F.data(), img)) setbit(G, x);
        if (g + 1 == (1u << n)) break;
        img ^= col[__builtin_ctz(g + 1)];
    }
    return G;
}

// ---- GF(2^n) power functions ------------------------------------------------
struct GF {
    int n = 2;
    u32 poly = 7;
    u32 mul(u32 a, u32 b) const {
        u32 r = 0;
        while (b) {
            if (b & 1) r ^= a;
            b >>= 1;
            a <<= 1;
            if (a >> n) a ^= poly;
        }
        return r;
    }
    u32 pow(u32 a, u64 e) const {
        u32 r = 1;
        while (e) {
            if (e & 1) r = mul(r, a);
            a = mul(a, a);
            e >>= 1;
        }
        return r;
    }
    u32 trace(u32 a) const {
        u32 t = 0;
        for (int i = 0; i < n; ++i) { t ^= a; a = mul(a, a); }
        return t & 1;
    }
};

static bool gf_for(int n, GF& g) {
    static const u32 polys[21] = {0, 0, 0x7, 0xB, 0x13, 0x25, 0x43, 0x83, 0x11D, 0x211,
                                  0x409, 0x805, 0x1053, 0x201B, 0x402B, 0x8003, 0x1100B,
                                  0x20009, 0x40081, 0x80027, 0x100009};
    if (n < 2 || n > 20) return false;
    g.n = n;
    g.poly = polys[n];
    return true;
}

// f(x) = Tr(alpha * x^d) with alpha = (generator)^e, generator = the element "x".
static TT power_function(int n, u64 d, u64 e) {
    GF g;
    gf_for(n, g);
    u32 alpha = g.pow(2, e);
    TT F(nwords(n), 0);
    for (u32 x = 0; x < (1u << n); ++x)
        if (g.trace(g.mul(alpha, g.pow(x, d)))) setbit(F, x);
    return F;
}

// Kasami bent function Tr(alpha x^(4^k - 2^k + 1)), alpha of order 3
// (n even, 3 does not divide n, gcd(k, n) = 1).  k = 3 gives the exponent 57.
static TT kasami(int n, int k = 3) {
    return power_function(n, (1ull << (2 * k)) - (1ull << k) + 1, ((1ull << n) - 1) / 3);
}

// Dillon's monomial bent function Tr(a x^(2^t - 1)), n = 2t, where a in F_(2^t)
// is a zero of the Kloosterman sum.  Returns an empty table if none is found.
static TT dillon(int n) {
    int t = n / 2;
    GF g;
    if (n % 2 || !gf_for(n, g)) return TT();
    u64 q = (1ull << t) - 1;
    u32 beta = g.pow(2, (1ull << t) + 1);  // generator of the subfield F_(2^t)^*
    auto tr_t = [&](u32 z) {               // trace of the subfield, computed in the big field
        u32 s = 0;
        for (int i = 0; i < t; ++i) { s ^= z; z = g.mul(z, z); }
        return s & 1;
    };
    std::vector<u32> pw(q);
    pw[0] = 1;
    for (u64 j = 1; j < q; ++j) pw[j] = g.mul(pw[j - 1], beta);
    for (u64 i = 1; i < q; ++i) {
        long long K = 1;  // the term y = 0
        for (u64 j = 0; j < q; ++j) K += tr_t(pw[(q - j) % q] ^ pw[(i + j) % q]) ? -1 : 1;
        if (K == 0) {
            TT F(nwords(n), 0);
            u32 a = pw[i];
            for (u32 x = 0; x < (1u << n); ++x)
                if (g.trace(g.mul(a, g.pow(x, q)))) setbit(F, x);
            return F;
        }
    }
    return TT();
}

// Direct sum with r hyperbolic pairs: f(x) + y1 y2 + ... + y(2r-1) y(2r), on n + 2r
// variables (x in the low n coordinates).  Bent iff f is bent.
static TT add_hyperbolic(const TT& F, int n, int r) {
    int m = n + 2 * r;
    TT G(nwords(m), 0);
    for (u32 z = 0; z < (1u << m); ++z) {
        u32 y = z >> n;
        int qv = 0;
        for (int i = 0; i < r; ++i) qv ^= (y >> (2 * i)) & (y >> (2 * i + 1)) & 1;
        if (getbit(F.data(), z & ((1u << n) - 1)) ^ qv) setbit(G, z);
    }
    return G;
}

// Is the element "x" a generator of the multiplicative group (polynomial primitive)?
static bool gf_primitive(int n) {
    GF g;
    if (!gf_for(n, g)) return false;
    u64 ord = (1ull << n) - 1, m = ord;
    for (u64 q = 2; q * q <= m; ++q)
        if (m % q == 0) {
            if (g.pow(2, ord / q) == 1) return false;
            while (m % q == 0) m /= q;
        }
    return m == 1 || g.pow(2, ord / m) != 1;
}

// f(x) = 1 iff wt(x) >= t  (majority: t = n/2 + 1).
static TT threshold_function(int n, int t) {
    TT F(nwords(n), 0);
    for (u32 x = 0; x < (1u << n); ++x)
        if (__builtin_popcount(x) >= t) setbit(F, x);
    return F;
}

// Hidden weighted bit function: f(x) = x_wt(x) (bits numbered from 1), f(0) = 0.
static TT hwbf(int n) {
    TT F(nwords(n), 0);
    for (u32 x = 1; x < (1u << n); ++x)
        if ((x >> (__builtin_popcount(x) - 1)) & 1) setbit(F, x);
    return F;
}

// Component <mask, S(x)> of the AES S-box (n = 8).
static TT aes_component(u32 mask) {
    GF g;
    g.n = 8;
    g.poly = 0x11B;
    TT F(nwords(8), 0);
    for (u32 x = 0; x < 256; ++x) {
        u32 y = x ? g.pow(x, 254) : 0, sb = y;
        for (int r = 1; r <= 4; ++r) sb ^= ((y << r) | (y >> (8 - r))) & 0xFF;
        sb ^= 0x63;
        if (__builtin_popcount(sb & mask) & 1) setbit(F, x);
    }
    return F;
}

// Dillon's partial-spread class PS_ap: f(x,y) = g(x / y) on F_(2^t)^2, g balanced, g(0) = 0.
static TT ps_ap(int n, u64 seed) {
    std::mt19937_64 rng(seed);
    int t = n / 2;
    GF g;
    gf_for(t, g);
    u32 T = 1u << t;
    std::vector<u32> idx;
    for (u32 i = 1; i < T; ++i) idx.push_back(i);
    for (size_t i = idx.size() - 1; i > 0; --i) std::swap(idx[i], idx[rng() % (i + 1)]);
    std::vector<unsigned char> gv(T, 0);
    for (u32 i = 0; i < T / 2; ++i) gv[idx[i]] = 1;
    TT F(nwords(n), 0);
    for (u32 y = 1; y < T; ++y) {
        u32 yi = g.pow(y, T - 2);
        for (u32 x = 0; x < T; ++x)
            if (gv[g.mul(x, yi)]) setbit(F, x | (y << t));
    }
    return F;
}

// Carlet-Feng function: support {0, 1, a, a^2, ..., a^(2^(n-1)-2)}, a primitive.
static TT carlet_feng(int n) {
    GF g;
    gf_for(n, g);
    TT F(nwords(n), 0);
    setbit(F, 0);
    u32 e = 1;
    for (u64 i = 0; i + 1 < (1ull << (n - 1)); ++i) { setbit(F, e); e = g.mul(e, 2); }
    return F;
}

// f(x) + <u, x>
static TT add_linear(const TT& F, int n, u32 u) {
    TT G = F;
    for (u32 x = 0; x < (1u << n); ++x)
        if (__builtin_popcount(x & u) & 1) G[x >> 6] ^= 1ULL << (x & 63);
    return G;
}

// Dual of a bent function: W_f(u) = 2^(n/2) (-1)^(dual(u)).  Empty if f is not bent.
static TT bent_dual(const TT& F, int n) {
    std::vector<int32_t> W((size_t)1 << n);
    for (size_t x = 0; x < W.size(); ++x) W[x] = getbit(F.data(), (u32)x) ? -1 : 1;
    wht(W.data(), n);
    TT G(nwords(n), 0);
    for (size_t u = 0; u < W.size(); ++u) {
        if ((int64_t)W[u] * W[u] != ((int64_t)1 << n)) return TT();
        if (W[u] < 0) setbit(G, (u32)u);
    }
    return G;
}

// max |Walsh coefficient|
static int64_t max_walsh(const TT& F, int n) {
    std::vector<int32_t> W((size_t)1 << n);
    for (size_t x = 0; x < W.size(); ++x) W[x] = getbit(F.data(), (u32)x) ? -1 : 1;
    wht(W.data(), n);
    int64_t mx = 0;
    for (int32_t w : W) mx = std::max<int64_t>(mx, w < 0 ? -w : w);
    return mx;
}

// ---- known symmetries of the constructions ----------------------------------
// These are candidates only: the solver checks every map against the truth
// table before it uses it, so a wrong guess costs nothing but the check.

static LMap to_lmap(const std::vector<u32>& col) {
    LMap M;
    for (int i = 0; i < 32; ++i) M.c[i] = i < (int)col.size() ? col[i] : 0;
    return M;
}
static AffMap identity_map(int n) {
    AffMap a;
    for (int i = 0; i < n; ++i) a.col.push_back(1u << i);
    return a;
}

// Maps x -> lambda * x^(2^i) of GF(2^n) that preserve f (or complement it):
// a generator of the group of multipliers lambda, and the smallest Frobenius
// powers that work together with a multiplier of order at most 2^(n/2) + 1.
static std::vector<AffMap> field_symmetries(const TT& F, int n) {
    std::vector<AffMap> out;
    GF g;
    if (F.size() != nwords(n) || !gf_for(n, g) || !gf_primitive(n)) return out;
    const u64 N = (1ull << n) - 1;
    std::mt19937_64 rng(777);
    const int NS = 48;  // a map must agree with f on these points before it gets the full check
    std::vector<u32> xs(NS);
    std::vector<std::vector<u32>> fr(n, std::vector<u32>(NS));  // fr[i][s] = xs[s]^(2^i)
    for (int s = 0; s < NS; ++s) {
        u32 y = xs[s] = (u32)(rng() % N) + 1;
        for (int i = 0; i < n; ++i) { fr[i][s] = y; y = g.mul(y, y); }
    }
    auto as_map = [&](int i, u32 lam) {
        AffMap a;
        for (int j = 0; j < n; ++j) {
            u32 y = 1u << j;
            for (int k = 0; k < i; ++k) y = g.mul(y, y);
            a.col.push_back(g.mul(lam, y));
        }
        return a;
    };
    auto valid = [&](int i, u32 lam) {
        bool same = true, flip = true;
        for (int s = 0; s < NS && (same || flip); ++s) {
            bool e = getbit(F.data(), g.mul(lam, fr[i][s])) == getbit(F.data(), xs[s]);
            same &= e;
            flip &= !e;
        }
        return (same || flip) && aut_kind(F.data(), n, to_lmap(as_map(i, lam).col), 0) >= 0;
    };
    std::vector<u64> divs(1, 1);  // divisors of 2^n - 1
    {
        u64 r = N;
        for (u64 q = 3; q * q <= r; q += 2)
            if (r % q == 0) {
                size_t base = divs.size();
                u64 pw = 1;
                while (r % q == 0) {
                    r /= q;
                    pw *= q;
                    for (size_t k = 0; k < base; ++k) divs.push_back(divs[k] * pw);
                }
            }
        if (r > 1) {
            size_t base = divs.size();
            for (size_t k = 0; k < base; ++k) divs.push_back(divs[k] * r);
        }
        std::sort(divs.begin(), divs.end(), [](u64 a, u64 b) { return a > b; });
    }
    // the multipliers that work form a cyclic group: find its order
    u64 r0 = 1;
    for (u64 r : divs) {
        if (r0 % r == 0) continue;
        if (valid(0, g.pow(2, N / r))) r0 = r0 / std::__gcd(r0, r) * r;
    }
    if (r0 > 1) out.push_back(as_map(0, g.pow(2, N / r0)));
    // Frobenius powers, with a multiplier of small order if needed
    std::vector<u32> lams(1, 1);
    {
        std::unordered_set<u32> seen;
        seen.insert(1);
        for (u64 r : divs) {
            if (r == 1 || r > (1ull << (n / 2)) + 1) continue;
            u32 h = g.pow(2, N / r), y = h;
            for (u64 k = 1; k < r; ++k) {
                if (seen.insert(y).second) lams.push_back(y);
                y = g.mul(y, h);
            }
        }
    }
    std::vector<int> found;
    for (int i = 1; i < n; ++i) {
        bool implied = false;
        for (int j : found) implied |= i % j == 0;
        if (implied) continue;
        for (u32 lam : lams)
            if (valid(i, lam)) { out.push_back(as_map(i, lam)); found.push_back(i); break; }
    }
    return out;
}

// Symmetric functions (f depends on wt(x) only): the coordinate transpositions.
// Majority-like functions have more: complementing every bit, and replacing
// x_0 by the parity of x, which extends S_n to S_(n+1).
static std::vector<AffMap> symmetric_symmetries(int n) {
    std::vector<AffMap> out;
    for (int i = 0; i + 1 < n; ++i) {
        AffMap a = identity_map(n);
        std::swap(a.col[i], a.col[i + 1]);
        out.push_back(a);
    }
    AffMap c = identity_map(n);
    c.t = n >= 32 ? ~0u : (1u << n) - 1;
    out.push_back(c);
    AffMap e = identity_map(n);
    for (int j = 1; j < n; ++j) e.col[j] ^= 1u;
    out.push_back(e);
    return out;
}

// PS_ap as built by ps_ap(): f(x, y) = g(x / y), invariant under (x, y) -> (cx, cy).
static std::vector<AffMap> ps_symmetries(int n) {
    std::vector<AffMap> out;
    int t = n / 2;
    GF g;
    if (n % 2 || !gf_for(t, g)) return out;
    AffMap a;
    a.col.resize(n);
    for (int j = 0; j < t; ++j) {
        u32 y = g.mul(2, 1u << j);
        a.col[j] = y;
        a.col[j + t] = y << t;
    }
    out.push_back(a);
    return out;
}

// Isometries of y1 y2 + y3 y4 + ... on the variables n0 .. n0+2r-1 (identity on the rest).
static void hyperbolic_symmetries(int n0, int r, std::vector<AffMap>& out) {
    int n = n0 + 2 * r;
    for (int i = 0; i < r; ++i) {  // y1 <-> y2 inside a pair
        AffMap a = identity_map(n);
        std::swap(a.col[n0 + 2 * i], a.col[n0 + 2 * i + 1]);
        out.push_back(a);
    }
    for (int i = 0; i + 1 < r; ++i) {
        int y1 = n0 + 2 * i, y2 = y1 + 1, y3 = y1 + 2, y4 = y1 + 3;
        AffMap a = identity_map(n);  // swap two pairs
        std::swap(a.col[y1], a.col[y3]);
        std::swap(a.col[y2], a.col[y4]);
        out.push_back(a);
        AffMap b = identity_map(n);  // (y1, y2, y3, y4) -> (y1 + y3, y2, y3, y4 + y2)
        b.col[y3] ^= 1u << y1;
        b.col[y2] ^= 1u << y4;
        out.push_back(b);
    }
}

// A function together with the symmetries its construction is known to have.
struct Spec {
    TT F;
    int n = -1;
    std::vector<AffMap> sym;
};
static bool g_known_sym = false;  // whether the constructions below fill in Spec::sym

static Spec sp_plain(TT F, int n) {
    Spec s;
    s.n = n;
    s.F = std::move(F);
    return s;
}
static Spec sp_field(TT F, int n) {  // a function of a field element in the built-in representation
    Spec s = sp_plain(std::move(F), n);
    if (g_known_sym && !s.F.empty()) s.sym = field_symmetries(s.F, n);
    return s;
}
static Spec sp_symmetric(TT F, int n) {
    Spec s = sp_plain(std::move(F), n);
    if (g_known_sym) s.sym = symmetric_symmetries(n);
    return s;
}
static Spec sp_ps(TT F, int n) {
    Spec s = sp_plain(std::move(F), n);
    if (g_known_sym) s.sym = ps_symmetries(n);
    return s;
}
// The dual of a bent function inherits the linear symmetries, transposed and inverted.
static Spec sp_dual(Spec s) {
    if (s.F.empty()) return s;
    s.F = bent_dual(s.F, s.n);
    std::vector<AffMap> out;
    for (const AffMap& a : s.sym) {
        LMap inv;
        if (!lm_inverse(to_lmap(a.col), s.n, inv)) continue;
        AffMap b;
        for (int i = 0; i < s.n; ++i) {
            u32 c = 0;
            for (int j = 0; j < s.n; ++j) c |= ((inv.c[j] >> i) & 1u) << j;
            b.col.push_back(c);
        }
        out.push_back(b);
    }
    s.sym = out;
    return s;
}
static Spec sp_plus(Spec s, u32 u) {  // the candidates stay; those that do not fix <u, x> fail the check
    if (!s.F.empty()) s.F = add_linear(s.F, s.n, u);
    return s;
}
static Spec sp_hyp(Spec s, int r) {
    if (s.F.empty() || r <= 0) return s;
    int n0 = s.n;
    s.F = add_hyperbolic(s.F, n0, r);
    s.n = n0 + 2 * r;
    for (AffMap& a : s.sym)
        for (int i = n0; i < s.n; ++i) a.col.push_back(1u << i);
    if (g_known_sym) hyperbolic_symmetries(n0, r, s.sym);
    return s;
}
static Spec sp_scramble(Spec s, u64 seed) {
    if (s.F.empty()) return s;
    std::vector<u32> col;
    u32 shift;
    scramble_map(s.n, seed, col, shift);
    s.F = scramble(s.F, s.n, seed);
    // G = F o T with T(x) = Mx + shift: an automorphism A of F becomes T^-1 A T
    LMap M = to_lmap(col), Mi;
    lm_inverse(M, s.n, Mi);
    for (AffMap& a : s.sym) {
        LMap A = to_lmap(a.col);
        LMap C = lm_mul(Mi, lm_mul(A, M, s.n), s.n);
        u32 t = lm_apply(Mi, lm_apply(A, shift) ^ a.t ^ shift);
        for (int i = 0; i < s.n; ++i) a.col[i] = C.c[i];
        a.t = t;
    }
    return s;
}
