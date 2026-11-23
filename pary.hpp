// The same question in characteristic p: the largest k such that
// f : F_p^n -> F_p (p prime) is constant on some k-dimensional flat.
//
// The method is the one of solver.hpp / search.hpp, rewritten for lines of p
// points instead of pairs:
//
//   1. invariant linear structures (f(x + v) = f(x) for all x) are factored out;
//   2. upper bounds from the Walsh spectrum, one per constant c: if f = c on
//      a + U, the transform B of the indicator of {f = c} satisfies
//      sum_{u in U^perp} B(u) w^(u.a) = p^n, hence
//      (p^(n-k) - 1) max_{u != 0} |B(u)| >= p^n - #{f = c};
//   3. a lower bound from a randomised greedy descent;
//   4. exhaustive decision searches for t = upper bound, upper bound - 1, ...
//      With maps under which f is invariant up to a permutation of its
//      values, fewer constants and fewer first directions are searched
//      (psym.hpp).
//
// The search picks the reduced echelon basis v_1, ..., v_j of the direction
// space in order of increasing leading coordinate (the highest non-zero one,
// with coefficient 1).  After picking v with leading coordinate pi it passes
// to the quotient by v: coordinate pi is deleted and the new set holds the
// lines x + F_p v that lie inside the old one.  Node = (T, m, j, lo): find a
// j-flat in T (a subset of F_p^m) whose directions have leading coordinate
// >= lo.  The next leading coordinate is at most m - j, and since dropping
// the coordinates below lo is injective on such a flat, at least p^j of the
// p^lo-blocks of T must be non-empty.
//
// A set is kept in one of two forms.
//   dense   one 64-bit cell per value of the low coordinates, holding the top
//           c coordinates (p^c <= 64 bits).  The directions tried at a node
//           never touch the top j - 1 coordinates, so with c <= j - 1 the
//           quotient is "AND p cells", the cells being found through a table.
//   sparse  a sorted list of points, each a 64-bit word with one bit field
//           per coordinate, so that vector addition is a few word operations.
//           All lines inside the set are found from the pairs (x, y) with
//           y - x a normalised direction; the coordinates are kept, a line
//           being represented by its point with a zero at the pivot.  The
//           pairs for pivot pi lie inside a group (the points that agree
//           above pi), where membership is a bit test in a small table; the
//           cells of a dense set are such a table, so the first step of this
//           kind works on them directly.  For
//           the largest pivot allowed, the other directions must have the
//           coordinates above it as pivots, so a direction needs a line in
//           every group: the groups are gone through one at a time and only
//           the directions that had a line in all the earlier ones are kept.
// A cost estimate chooses between them at every node.
//
// Points and vectors are reported as integers x = x_0 + x_1 p + ... ; every
// flat is re-checked point by point against the table.
#pragma once
#include "solver.hpp"
#include <complex>

typedef std::vector<uint8_t> PT;  // f[x], x = x_0 + x_1 p + ... + x_(n-1) p^(n-1)
typedef std::complex<double> cplx;

static inline u64 pext64(u64 x, u64 m) {
#ifdef __BMI2__
    return _pext_u64(x, m);
#else
    u64 r = 0;
    for (int k = 0; m; m &= m - 1, ++k) r |= ((x >> ctz64(m)) & 1) << k;
    return r;
#endif
}

template <class Fn> static void pa_parallel(int nt, Fn fn) {
    std::vector<std::thread> th;
    for (int i = 1; i < nt; ++i) th.emplace_back(fn, i);
    fn(0);
    for (auto& t : th) t.join();
}

// Arithmetic of F_p^m on packed vectors: digit i sits in bits [i B, (i+1) B),
// and the top bit of every field is free, so that sums do not carry over.
struct PField {
    int p = 0;
    int B = 0;        // bits per digit, 2^(B-1) >= p
    int c0 = 0;       // largest c with p^c <= 64
    int maxdim = 0;   // digits in a packed vector
    u64 ones = 0, red = 0, pall = 0, dmask = 0;
    u64 pw[41];       // p^i (saturating)
    u64 packb[64];    // packed digits of a position inside a cell
    u64 dsel[7][64];  // dsel[c][s]: positions of a c-digit cell whose lowest digit is s
    int inv[64];
    int cd = 0;       // digits looked up at a time by index()
    u64 cmask = 0;
    u32 ctab[1024];   // ctab[v] = the index of the cd digits packed in v

    static bool prime_ok(int q) {
        if (q < 2 || q > 61) return false;
        for (int d = 2; d * d <= q; ++d)
            if (q % d == 0) return false;
        return true;
    }
    bool init(int q) {
        if (!prime_ok(q)) return false;
        p = q;
        B = 2;
        while ((1 << (B - 1)) < p) ++B;
        maxdim = 64 / B;
        dmask = (1ULL << B) - 1;
        ones = 0;
        for (int i = 0; i < maxdim; ++i) ones |= 1ULL << (i * B);
        red = ones * ((1ULL << (B - 1)) - (u64)p);
        pall = ones * (u64)p;
        pw[0] = 1;
        for (int i = 1; i <= 40; ++i) pw[i] = pw[i - 1] > ~0ULL / p ? ~0ULL : pw[i - 1] * p;
        c0 = 0;
        while (pw[c0 + 1] <= 64) ++c0;
        for (u64 b = 0; b < 64; ++b) packb[b] = pack(b, c0);
        memset(dsel, 0, sizeof dsel);
        for (int c = 1; c <= c0; ++c)
            for (int s = 0; s < p; ++s)
                for (u64 i = 0; i < pw[c - 1]; ++i) dsel[c][s] |= 1ULL << (s + p * i);
        for (int a = 1; a < p; ++a)
            for (int b = 1; b < p; ++b)
                if (a * b % p == 1) inv[a] = b;
        cd = std::max(1, 10 / B);
        cmask = (1ULL << (cd * B)) - 1;
        for (u64 v = 0; v <= cmask; ++v) {
            u64 r = 0;
            for (int i = cd - 1; i >= 0; --i) r = r * p + std::min<u64>((v >> (i * B)) & dmask, p - 1);
            ctab[v] = (u32)r;
        }
        return true;
    }
    u64 index(u64 x, int nd) const {  // the index of a packed vector with nd digits
        u64 r = 0;
        for (int i = 0, sh = 0; i < nd; i += cd, sh += cd * B) r += (u64)ctab[(x >> sh) & cmask] * pw[i];
        return r;
    }
    int cdim(int j) const { return std::min(c0, j - 1); }  // cell dimension of a dense node with j directions to go
    u64 add(u64 x, u64 y) const {
        const u64 s = x + y;
        return s - (((s + red) >> (B - 1)) & ones) * (u64)p;
    }
    u64 sub(u64 x, u64 y) const {
        const u64 s = x + pall - y;
        return s - (((s + red) >> (B - 1)) & ones) * (u64)p;
    }
    u64 scale(u64 x, int s) const {
        u64 r = 0;
        for (; s > 0; --s) r = add(r, x);
        return r;
    }
    int digit(u64 x, int i) const { return (int)((x >> (i * B)) & dmask); }
    u64 above(u64 x, int i) const { return (x >> (i * B)) >> B; }  // the digits above digit i
    int lead(u64 x) const { return (63 - __builtin_clzll(x)) / B; }  // highest non-zero digit, x != 0
    u64 pack(u64 idx, int m) const {
        u64 r = 0;
        for (int i = 0; i < m; ++i, idx /= p) r |= (idx % p) << (i * B);
        return r;
    }
    u64 unpack(u64 x, int m) const {
        u64 r = 0;
        for (int i = m - 1; i >= 0; --i) r = r * p + digit(x, i);
        return r;
    }
    u64 next(u64 x) const {  // the packed vector of the next index
        x += 1;
        for (int i = 0; i + 1 < maxdim && digit(x, i) == p; ++i) x += (1ULL << ((i + 1) * B)) - ((u64)p << (i * B));
        return x;
    }
};

// x + v in F_p^n on indices (slow; used outside the search only).
static u32 pa_addidx(int p, int n, u32 x, u32 v) {
    u32 r = 0, w = 1;
    for (int i = 0; i < n; ++i, x /= p, v /= p, w *= p) r += (x % p + v % p) % p * w;
    return r;
}

// a[u] <- sum_x a[x] w^(-<u, x>) over F_p^n, w = exp(2 pi i / p).
static void pa_fft(cplx* a, int p, int n, const u64* pw, int threads) {
    const double PI2 = 6.283185307179586476925286766559;
    std::vector<cplx> w(p);
    for (int k = 0; k < p; ++k) w[k] = std::polar(1.0, -PI2 * k / p);
    const size_t NF = pw[n] / p;
    for (int i = 0; i < n; ++i) {
        const size_t s = pw[i];
        const int nt = NF >= 8192 ? std::max(1, threads) : 1;
        pa_parallel(nt, [&](int ti) {
            std::vector<cplx> t(p), v(p);
            const size_t lo = NF * ti / nt, hi = NF * (ti + 1) / nt;
            for (size_t q = lo; q < hi; ++q) {
                cplx* b = a + (q / s) * s * p + q % s;
                for (int k = 0; k < p; ++k) v[k] = b[k * s];
                for (int u = 0; u < p; ++u) {
                    cplx acc = v[0];
                    for (int k = 1, idx = 0; k < p; ++k) {
                        idx += u;
                        if (idx >= p) idx -= p;
                        acc += v[k] * w[idx];
                    }
                    t[u] = acc;
                }
                for (int u = 0; u < p; ++u) b[u * s] = t[u];
            }
        });
    }
}

struct PSpec {
    double max_walsh = 0;    // max_u |sum_x w^(f(x) - <u, x>)|
    bool bent = false;
    std::vector<int> ub;     // per constant c: no flat with f = c has a larger dimension
    std::vector<u64> size;   // #{f = c}
};

static PSpec pa_spectrum(const PField& F, const uint8_t* f, int n, int threads) {
    const int p = F.p;
    const size_t N = F.pw[n];
    PSpec S;
    S.size.assign(p, 0);
    S.ub.assign(p, -1);
    for (size_t x = 0; x < N; ++x) ++S.size[f[x]];
    const double PI2 = 6.283185307179586476925286766559;
    std::vector<cplx> w(p);
    for (int k = 0; k < p; ++k) w[k] = std::polar(1.0, PI2 * k / p);
    // V_t(u) = sum_x w^(t f(x) - <u, x>) for t = 1 .. H; V_(p-t)(u) is the conjugate of V_t(-u).
    const int H = p == 2 ? 1 : (p - 1) / 2;
    const bool full = (double)H * 16.0 * (double)N <= 1.5e9 && (double)N * p * p <= 8e9;
    const int nv = full ? H : 1;
    std::vector<std::vector<cplx>> V(nv);
    for (int t = 1; t <= nv; ++t) {
        V[t - 1].resize(N);
        for (size_t x = 0; x < N; ++x) V[t - 1][x] = w[(t * f[x]) % p];
        pa_fft(V[t - 1].data(), p, n, F.pw, threads);
    }
    double hi2 = 0, lo2 = 1e300;
    for (size_t u = 0; u < N; ++u) {
        const double a = std::norm(V[0][u]);
        hi2 = std::max(hi2, a);
        lo2 = std::min(lo2, a);
    }
    S.max_walsh = std::sqrt(hi2);
    S.bent = std::fabs(hi2 - (double)N) <= 1e-6 * N && std::fabs(lo2 - (double)N) <= 1e-6 * N;
    int kw = 0;  // p^k <= max |V_1|
    while (kw < n && (double)F.pw[kw + 1] <= S.max_walsh * (1 + 1e-9) + 1e-6) ++kw;
    std::vector<double> M(p, 0.0);  // max over u != 0 of |B_c(u)|, B_c the transform of the indicator of {f = c}
    if (full && N > 1) {
        const int nt = N >= 8192 ? std::max(1, threads) : 1;
        std::mutex mu;
        pa_parallel(nt, [&](int ti) {
            std::vector<double> m(p, 0.0);
            std::vector<cplx> val(p);
            const size_t lo = 1 + (N - 1) * ti / nt, hi = 1 + (N - 1) * (ti + 1) / nt;
            for (size_t u = lo; u < hi; ++u) {
                size_t nu = 0, r = u, wgt = 1;
                for (int i = 0; i < n; ++i, r /= p, wgt *= p) nu += (p - r % p) % p * wgt;
                for (int t = 1; t < p; ++t) val[t] = t <= H ? V[t - 1][u] : std::conj(V[p - t - 1][nu]);
                for (int c = 0; c < p; ++c) {
                    cplx b = 0;
                    for (int t = 1; t < p; ++t) b += val[t] * w[(p - t * c % p) % p];
                    m[c] = std::max(m[c], std::norm(b));
                }
            }
            std::lock_guard<std::mutex> g(mu);
            for (int c = 0; c < p; ++c) M[c] = std::max(M[c], m[c]);
        });
        for (int c = 0; c < p; ++c) M[c] = std::sqrt(M[c]) / p;
    }
    for (int c = 0; c < p; ++c) {
        if (!S.size[c]) continue;
        int k = kw;
        while (k > 0 && F.pw[k] > S.size[c]) --k;
        if (full)
            while (k > 0 && ((double)F.pw[n - k] - 1) * M[c] * (1 + 1e-9) + 1e-6 < (double)N - (double)S.size[c]) --k;
        S.ub[c] = k;
    }
    return S;
}

// The points of a sparse node, for membership tests.
struct PHash {
    std::vector<u64> k;  // point + 1, 0 = empty
    u64 mask = 0;
    int shift = 0;
    void build(const u64* L, size_t N) {
        size_t cap = 16;
        int lg = 4;
        while (cap < 2 * N) cap <<= 1, ++lg;
        if (k.size() < cap) k.resize(cap);
        std::fill(k.begin(), k.begin() + cap, 0);
        mask = cap - 1;
        shift = 64 - lg;
        for (size_t i = 0; i < N; ++i) {
            const u64 key = L[i] + 1;
            size_t h = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> shift);
            while (k[h]) h = (h + 1) & mask;
            k[h] = key;
        }
    }
    bool has(u64 z) const {
        const u64 key = z + 1;
        for (size_t h = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> shift); k[h]; h = (h + 1) & mask)
            if (k[h] == key) return true;
        return false;
    }
};

// The quotient of a dense set by v = vlow + e_pi.  The cells of the set come
// in superblocks of p * P cells (P = p^pi, one superblock per value of the
// cell coordinates above pi); tr[(t-1) P + low] is the place, inside a
// superblock, of the cell (low + t vlow, t).  Returns the number of non-empty
// blocks of the quotient (block = the points that agree in the coordinates
// >= pi), giving up, with a smaller number, once `need` is out of reach.
static u64 pa_quot(int p, const u64* in, u64* out, size_t P, size_t nhigh, const u32* tr, u64 need, u64 cellbits) {
    u64 blocks = 0;
    const size_t SB = P * p;
    for (size_t h = 0; h < nhigh; ++h, in += SB, out += P) {
        u64 orr = 0;
        if (p == 3) {
            const u32 *t1 = tr, *t2 = tr + P;
            for (size_t x = 0; x < P; ++x) {
                const u64 a = in[x] & in[t1[x]] & in[t2[x]];
                out[x] = a;
                orr |= a;
            }
        } else if (p == 2) {
            for (size_t x = 0; x < P; ++x) {
                const u64 a = in[x] & in[tr[x]];
                out[x] = a;
                orr |= a;
            }
        } else {
            for (size_t x = 0; x < P; ++x) out[x] = in[x] & in[tr[x]];
            for (int t = 2; t < p; ++t) {
                const u32* tt = tr + (size_t)(t - 1) * P;
                for (size_t x = 0; x < P; ++x) out[x] &= in[tt[x]];
            }
            for (size_t x = 0; x < P; ++x) orr |= out[x];
        }
        blocks += popc(orr);
        if (blocks + (u64)(nhigh - 1 - h) * cellbits < need) return blocks;
    }
    return blocks;
}

struct PFlat {
    int c = 0;
    u32 a = 0;
    std::vector<u32> basis;
};

static size_t pa_rec_limit = (size_t)1 << 22;  // lines of one pass kept in memory at a time (lowered by the tests)

struct PCtx {
    // Scratch space of the node at one depth (depth = directions chosen so far).
    struct Sp {
        struct Grp { u32 i, a1, a2, e; };  // [i, a1): digit 0 at the pivot, [a1, a2): digit 1, [a2, e): the rest
        std::vector<u64> list, kids, rpt, mkey, sdir, bm, a0, a1;
        std::vector<u32> rec, mval, scnt, soff, viable, stamp, alive, gidx;
        std::vector<std::vector<u64>> b0, b1;
        std::vector<Grp> grp;
        std::vector<std::pair<u32, u32>> todo;
        size_t mcap = 0;
        int mshift = 0;
        static u64 hash(u64 dv) { return dv * 0x9E3779B97F4A7C15ULL; }
        void map_clear() {
            mcap = 64;
            mshift = 58;
            if (mkey.size() < 64) mkey.resize(64), mval.resize(64);
            std::fill(mkey.begin(), mkey.begin() + 64, 0);
            sdir.clear();
            scnt.clear();
            stamp.clear();
        }
        u32 find(u64 dv) const {  // the number of a direction seen before, or ~0
            for (size_t h = (size_t)(hash(dv) >> mshift); mkey[h]; h = (h + 1) & (mcap - 1))
                if (mkey[h] == dv) return mval[h];
            return ~0u;
        }
        void grow() {
            mcap *= 2;
            --mshift;
            if (mkey.size() < mcap) mkey.resize(mcap), mval.resize(mcap);
            std::fill(mkey.begin(), mkey.begin() + mcap, 0);
            for (u32 s = 0; s < sdir.size(); ++s) {
                size_t h = (size_t)(hash(sdir[s]) >> mshift);
                while (mkey[h]) h = (h + 1) & (mcap - 1);
                mkey[h] = sdir[s];
                mval[h] = s;
            }
        }
        u32 slot(u64 dv) {  // the number of this direction; counts its lines
            size_t h = (size_t)(hash(dv) >> mshift);
            for (; mkey[h]; h = (h + 1) & (mcap - 1))
                if (mkey[h] == dv) {
                    ++scnt[mval[h]];
                    return mval[h];
                }
            const u32 s = (u32)sdir.size();
            mkey[h] = dv;
            mval[h] = s;
            sdir.push_back(dv);
            scnt.push_back(1);
            stamp.push_back(0);
            if (2 * sdir.size() > mcap) grow();
            return s;
        }
    };
    const PField* F = nullptr;
    const Options* opt = nullptr;
    std::atomic<bool>* stop = nullptr;
    // 0: cost estimate, 1: sparse wherever possible, 2: dense wherever possible,
    // 3: as 0, but every sparse step on a dense set is given up at once (for the tests)
    int mode = 0;
    Stats st;
    std::vector<std::vector<u64>> buf, tmp;
    std::vector<std::vector<u32>> trv;
    std::vector<Sp> sp;
    // The path to the current node: ddepth dense steps (pivot and low part of
    // the direction, in the coordinates of their level), then sparse steps
    // (packed vectors in the sdim coordinates of the level where the sparse
    // part began), and a point of the last set.
    int ddepth = 0, sdim = 0;
    int dpiv[48];
    u32 dlow[48];
    u64 pdir[48];
    u64 leaf = 0;
    // If set: called at the root in place of the loop over the children of one pass.
    std::function<bool(Sp&, int, int)> kid_hook;
    // If set: bit v says that the first direction v (as an index; leading coordinate <= roothi) is not to be tried.
    const std::vector<u64>* rootskip = nullptr;
    int roothi = 0;

    void init(const PField* f, int n, const Options* o, std::atomic<bool>* s, int md) {
        F = f;
        opt = o;
        stop = s;
        mode = md;
        buf.resize(n + 2);
        tmp.resize(n + 2);
        trv.resize(n + 2);
        sp.resize(n + 2);
    }
    bool stopped() const { return stop->load(std::memory_order_relaxed); }

    // ---------------------------------------------------------------- dense
    void build_tr(u32* tr, int pi, u32 vlow) const {
        const int p = F->p;
        const size_t P = F->pw[pi];
        int dg[48];
        for (int i = 0; i < pi; ++i, vlow /= p) dg[i] = vlow % p;
        for (int t = 1; t < p; ++t) {
            u32* T = tr + (size_t)(t - 1) * P;
            T[0] = (u32)(t * P);
            size_t size = 1;
            for (int i = 0; i < pi; ++i, size *= p) {
                const u32 s = (u32)(t * dg[i] % p);
                for (int u = 1; u < p; ++u) {
                    const u32 off = (u32)((u + s) % p * size);
                    u32* D = T + u * size;
                    for (size_t x = 0; x < size; ++x) D[x] = T[x] + off;
                }
                if (s) {
                    const u32 off = (u32)(s * size);
                    for (size_t x = 0; x < size; ++x) T[x] += off;
                }
            }
        }
    }
    // Cost estimates of one node, in cell operations.
    double dense_cost(int m, int j, int lo) const {
        const int hi = m - j;
        if (hi < lo) return 0;
        const double cand = ((double)F->pw[hi + 1] - (double)F->pw[lo]) / (F->p - 1);
        return cand * (double)F->pw[m - F->cdim(j)];
    }
    double pair_weight() const { return opt->sparse_weight; }
    double sparse_cost(u64 N, int m, int j, int lo) const {
        double s = 0;
        for (int pi = lo; pi <= m - j; ++pi) s += 1.0 / (double)F->pw[m - pi + 1];
        return pair_weight() * (double)N * (double)N * s + 16.0 * (double)N;
    }
    // Sorted list of the points of a dense set.
    void extract(const u64* T, int m, int cin, std::vector<u64>& L) const {
        const int mc = m - cin, B = F->B;
        const size_t ncell = F->pw[mc];
        size_t cnt[65];
        memset(cnt, 0, sizeof cnt);
        for (size_t i = 0; i < ncell; ++i)
            for (u64 w = T[i]; w; w &= w - 1) ++cnt[ctz64(w) + 1];
        for (int b = 0; b < 64; ++b) cnt[b + 1] += cnt[b];
        L.resize(cnt[64]);
        u64 low = 0;
        for (size_t i = 0; i < ncell; ++i, low = F->next(low))
            for (u64 w = T[i]; w; w &= w - 1) {
                const int b = ctz64(w);
                L[cnt[b]++] = low | (F->packb[b] << (mc * B));
            }
    }
    // The same set with cells of one coordinate less.
    void relayout(const u64* T, int m, int cin, std::vector<u64>& out) const {
        const int p = F->p;
        const size_t ncell = F->pw[m - cin];
        if (out.size() < ncell * p) out.resize(ncell * p);
        for (int s = 0; s < p; ++s) {
            const u64 sel = F->dsel[cin][s];
            u64* o = out.data() + (size_t)s * ncell;
            for (size_t i = 0; i < ncell; ++i) o[i] = pext64(T[i], sel);
        }
    }
    // Where the points of a node are: a sorted list, or the cells of a dense set.
    struct Src {
        const u64* L = nullptr;
        size_t N = 0;
        const u64* T = nullptr;
        int c = 0, mc = 0;
    };
    // A node whose set is in dense form with cells of cin coordinates.
    bool enter(int d, const u64* T, int m, int j, int lo, int cin) {
        if (j == 1) {  // any line will do: from the list of points
            Sp& S = sp[d];
            extract(T, m, cin, S.list);
            ddepth = d;
            sdim = m;
            return sparse(d, S.list.data(), S.list.size(), 1, lo);
        }
        const int c = F->cdim(j);
        if (c != cin) {
            relayout(T, m, cin, buf[d]);
            T = buf[d].data();
        }
        if (mode == 1 || (mode != 2 && sparse_cost(popcount_arr(T, F->pw[m - c]), m, j, lo) < dense_cost(m, j, lo))) {
            // by pairs, straight from the cells; given up if the points turn out to be
            // so unevenly spread that it costs more than the dense step
            ++st.sparse_nodes;
            ddepth = d;
            sdim = m;
            Src src;
            src.T = T;
            src.c = c;
            src.mc = m - c;
            const u64 budget = mode == 1 ? ~0ULL : mode == 3 ? 0 : (u64)(4 * dense_cost(m, j, lo) / pair_weight()) + 100000;
            const int r = pivots(d, src, j, lo, budget);
            if (r != 3) return r == 1;
        }
        return dense(d, T, m, j, lo);
    }
    bool dense(int d, const u64* T, int m, int j, int lo) {  // j >= 2
        ++st.dense_nodes;
        const int p = F->p, c = F->cdim(j), mc = m - c, hi = m - j;
        const u64 need = F->pw[j - 1], cellbits = F->pw[c];
        const size_t ncell = F->pw[mc - 1];
        std::vector<u64>& ob = tmp[d + 1];
        if (ob.size() < ncell) ob.resize(ncell);
        for (int pi = hi; pi >= lo; --pi) {  // the largest pivot first: most flats have it
            const size_t P = F->pw[pi], nhigh = F->pw[mc - 1 - pi];
            if (trv[d].size() < (size_t)(p - 1) * P) trv[d].resize((size_t)(p - 1) * P);
            for (u32 vl = 0; vl < P; ++vl) {
                if (stopped()) return false;
                ++st.cands;
                st.work += (u64)p * ncell;
                build_tr(trv[d].data(), pi, vl);
                if (pa_quot(p, T, ob.data(), P, nhigh, trv[d].data(), need, cellbits) < need) continue;
                dpiv[d] = pi;
                dlow[d] = vl;
                if (enter(d + 1, ob.data(), m - 1, j - 1, pi, c)) return true;
            }
        }
        return false;
    }

    // --------------------------------------------------------------- sparse
    // Runs [i, e) of points that agree above coordinate pi: [i, a1) have digit 0 there, [a1, a2) digit 1.
    template <class Fn> void for_groups(const u64* L, size_t N, int pi, Fn fn) const {
        for (size_t i = 0; i < N;) {
            const u64 key = F->above(L[i], pi);
            size_t e = i + 1;
            while (e < N && F->above(L[e], pi) == key) ++e;
            size_t a1 = i;
            while (a1 < e && F->digit(L[a1], pi) == 0) ++a1;
            size_t a2 = a1;
            while (a2 < e && F->digit(L[a2], pi) == 1) ++a2;
            if (a1 > i && a2 > a1 && fn(i, a1, a2, e)) return;
            i = e;
        }
    }
    u64 blocks_above(const u64* L, size_t N, int lo) const {  // distinct values of the coordinates >= lo
        u64 c = 1;
        const int sh = lo * F->B;
        for (size_t i = 1; i < N; ++i) c += (L[i] >> sh) != (L[i - 1] >> sh);
        return c;
    }
    // A node whose set is the sorted list L, in the sdim coordinates of the sparse part.
    bool sparse(int d, const u64* L, size_t N, int j, int lo) {
        ++st.sparse_nodes;
        if (N == 0) return false;
        if (j == 0) {
            leaf = L[0];
            return true;
        }
        if (lo > sdim - j || N < F->pw[j]) return false;
        if (lo > 0 && blocks_above(L, N, lo) < F->pw[j]) return false;
        Src src;
        src.L = L;
        src.N = N;
        return pivots(d, src, j, lo, ~0ULL) == 1;
    }
    // All pivots of a node, the largest first.  1: flat found, 3: more than `budget` pairs, 0: nothing.
    int pivots(int d, const Src& src, int j, int lo, u64 budget) {
        Sp& S = sp[d];
        const u64 wlimit = budget == ~0ULL ? ~0ULL : st.work + budget;
        for (int pi = sdim - j; pi >= lo; --pi) {
            // one pass per pivot; if it finds too many lines the directions are split into classes
            S.todo.clear();
            S.todo.push_back({1, 0});
            while (!S.todo.empty()) {
                if (stopped()) return 0;
                const std::pair<u32, u32> cls = S.todo.back();
                S.todo.pop_back();
                const int r = pass(d, src, j, pi, cls.first, cls.second, wlimit);
                if (r == 1 || r == 3) return r;
                if (r == 2)
                    for (u32 q = 0; q < 4; ++q) S.todo.push_back({4 * cls.first, cls.second + q * cls.first});
            }
        }
        return 0;
    }
    // The directions with pivot pi (those of class g modulo G).
    // 1: flat found, 2: too many lines at once, 3: over the work limit, 0: nothing.
    int pass(int d, const Src& src, int j, int pi, u32 G, u32 g, u64 wlimit) {
        Sp& S = sp[d];
        const int p = F->p, B = F->B, nd = pi + 1;
        const u64 need = F->pw[j - 1], P = F->pw[pi], one = 1ULL << (pi * B);
        const u64 lowmask = nd * B >= 64 ? ~0ULL : (1ULL << (nd * B)) - 1;
        const bool cells = src.T != nullptr;
        // With the largest pivot the other directions have the coordinates above it as pivots,
        // so a direction needs a line in every one of the p^(j-1) groups.
        const bool top = j >= 2 && pi == sdim - j;
        size_t ngroups, nb = 1;
        if (cells) {
            nb = F->pw[src.c];
            ngroups = F->pw[src.mc - 1 - pi] * nb;
            if (!top && S.b0.size() < 64) S.b0.resize(64), S.b1.resize(64);
        } else {
            S.grp.clear();
            for_groups(src.L, src.N, pi, [&](size_t i, size_t a1, size_t a2, size_t e) {
                S.grp.push_back({(u32)i, (u32)a1, (u32)a2, (u32)e});
                return false;
            });
            ngroups = S.grp.size();
            if (top)  // cheapest group first
                std::sort(S.grp.begin(), S.grp.end(), [](const Sp::Grp& x, const Sp::Grp& y) {
                    return (u64)(x.a1 - x.i) * (x.a2 - x.a1) < (u64)(y.a1 - y.i) * (y.a2 - y.a1);
                });
            if (S.bm.size() < F->pw[nd] / 64 + 1) S.bm.resize(F->pw[nd] / 64 + 1, 0);
        }
        if (!ngroups || (top && ngroups < need)) return 0;
        // Membership in the current group, by the index of the digits 0 .. pi of a point:
        // bit wbit of the cell W[index], or bit `index` of the table W.
        const u64* W = S.bm.data();
        int wbit = 0;
        auto inset = [&](u64 z) {
            const u64 ix = F->index(z & lowmask, nd);
            return cells ? (W[ix] >> wbit) & 1 : (W[ix >> 6] >> (ix & 63)) & 1;
        };
        auto rest = [&](u64 y, u64 dv) {  // x and y = x + dv are in the set; are x + 2 dv, ... ?
            for (int t = 2; t < p; ++t) {
                y = F->add(y, dv);
                if (!inset(y)) return false;
            }
            return true;
        };
        S.map_clear();
        S.rec.clear();
        S.rpt.clear();
        bool over = false, found = false;
        auto line = [&](u32 s, u32 round, u64 x) {  // a line of direction number s through x
            S.stamp[s] = round;
            S.rec.push_back(s);
            S.rpt.push_back(x);
            if (S.rec.size() > pa_rec_limit && S.sdir.size() > 1 && G < (1u << 20)) over = true;
        };
        u32 round = 0;
        for (size_t gi = 0; gi < ngroups && !found && !over; ++gi) {
            const u64 *A0, *A1;  // the points of the group with digit 0 and 1 at the pivot
            size_t n0, n1;
            u32 gbeg = 0, gend = 0;
            if (cells) {
                const size_t h = gi / nb;
                const int b = (int)(gi % nb);
                const u64* base = src.T + h * P * p;
                const u64 hpart = F->pack(h, src.mc - 1 - pi) << (pi * B) << B;
                if (top) {  // few groups are looked at: this one only
                    S.a0.clear();
                    S.a1.clear();
                    u64 pl = hpart | (F->packb[b] << (src.mc * B));
                    for (size_t low = 0; low < P; ++low, pl = F->next(pl)) {
                        if ((base[low] >> b) & 1) S.a0.push_back(pl);
                        if ((base[P + low] >> b) & 1) S.a1.push_back(pl | one);
                    }
                    A0 = S.a0.data(), n0 = S.a0.size();
                    A1 = S.a1.data(), n1 = S.a1.size();
                } else {
                    if (b == 0) {  // all the groups in these cells at once
                        for (size_t q = 0; q < nb; ++q) S.b0[q].clear(), S.b1[q].clear();
                        u64 pl = hpart;
                        for (size_t low = 0; low < P; ++low, pl = F->next(pl)) {
                            for (u64 w = base[low]; w; w &= w - 1) {
                                const int q = ctz64(w);
                                S.b0[q].push_back(pl | (F->packb[q] << (src.mc * B)));
                            }
                            for (u64 w = base[P + low]; w; w &= w - 1) {
                                const int q = ctz64(w);
                                S.b1[q].push_back(pl | one | (F->packb[q] << (src.mc * B)));
                            }
                        }
                    }
                    A0 = S.b0[b].data(), n0 = S.b0[b].size();
                    A1 = S.b1[b].data(), n1 = S.b1[b].size();
                }
                W = base;
                wbit = b;
            } else {
                const Sp::Grp& gr = S.grp[gi];
                A0 = src.L + gr.i, n0 = gr.a1 - gr.i;
                A1 = src.L + gr.a1, n1 = gr.a2 - gr.a1;
                gbeg = gr.i, gend = gr.e;
            }
            if (!n0 || !n1) {
                if (top) return 0;
                continue;
            }
            ++round;
            if (top && round > 1) {  // the directions with a line in every group so far
                S.alive.clear();
                for (u32 s = 0; s < S.stamp.size(); ++s)
                    if (S.stamp[s] == round - 1) S.alive.push_back(s);
                if (S.alive.empty()) return 0;
            }
            if (!cells && p > 2) {
                S.gidx.resize(gend - gbeg);
                for (u32 q = gbeg; q < gend; ++q) {
                    const u64 ix = F->index(src.L[q] & lowmask, nd);
                    S.gidx[q - gbeg] = (u32)ix;
                    S.bm[ix >> 6] |= 1ULL << (ix & 63);
                }
            }
            if (top && round > 1 && S.alive.size() <= n1) {
                st.work += (u64)n0 * S.alive.size();
                for (u32 s : S.alive) {
                    const u64 dv = S.sdir[s];
                    for (size_t xi = 0; xi < n0 && !over; ++xi) {
                        const u64 y = F->add(A0[xi], dv);
                        if (p > 2 ? inset(y) && rest(y, dv) : std::binary_search(A1, A1 + n1, y)) {
                            ++S.scnt[s];
                            line(s, round, A0[xi]);
                        }
                    }
                }
            } else {
                st.work += (u64)n0 * n1;
                for (size_t xi = 0; xi < n0 && !over && !found; ++xi) {
                    const u64 x = A0[xi];
                    for (size_t yi = 0; yi < n1; ++yi) {
                        const u64 y = A1[yi], dv = F->sub(y, x);
                        if (top && round > 1) {  // only the directions still alive
                            const u32 s = S.find(dv);
                            if (s == ~0u || S.stamp[s] < round - 1 || !rest(y, dv)) continue;
                            ++S.scnt[s];
                            line(s, round, x);
                        } else {
                            if (G > 1 && (u32)((dv * 0xD6E8FEB86659FD93ULL) >> 40 & (G - 1)) != g) continue;
                            if (!rest(y, dv)) continue;
                            if (j == 1) {
                                pdir[d] = dv;
                                leaf = x;
                                found = true;
                                break;
                            }
                            line(S.slot(dv), round, x);
                        }
                        if (over) break;
                    }
                }
            }
            if (!cells && p > 2)
                for (u32 q = 0; q < gend - gbeg; ++q) S.bm[S.gidx[q] >> 6] = 0;
            if (cells && st.work > wlimit) return 3;
        }
        if (found) return 1;
        if (over) return 2;
        // children: the directions with enough lines, each with the sorted list of its lines
        const u32 nslot = (u32)S.sdir.size();
        S.soff.resize(nslot);
        S.viable.clear();
        u32 total = 0;
        for (u32 s = 0; s < nslot; ++s) {
            bool take = top ? S.stamp[s] == round : S.scnt[s] >= need;
            if (take && d == 0 && rootskip && !cells) {
                const u64 v = F->index(S.sdir[s], roothi + 1);
                if (((*rootskip)[v >> 6] >> (v & 63)) & 1) take = false, ++st.sym_skipped;
            }
            if (take) {
                S.soff[s] = total;
                total += S.scnt[s];
                S.viable.push_back(s);
            } else S.soff[s] = ~0u;
        }
        if (S.viable.empty()) return 0;
        if (S.kids.size() < total) S.kids.resize(total);
        for (size_t r = 0; r < S.rec.size(); ++r) {
            const u32 s = S.rec[r];
            if (S.soff[s] != ~0u) S.kids[S.soff[s]++] = S.rpt[r];
        }
        for (u32 s : S.viable)  // soff[s] is now the end of the list of s
            std::sort(S.kids.begin() + (S.soff[s] - S.scnt[s]), S.kids.begin() + S.soff[s]);
        st.cands += S.viable.size();
        if (d == 0 && kid_hook) return kid_hook(S, j, pi) ? 1 : 0;
        for (u32 s : S.viable) {
            pdir[d] = S.sdir[s];
            if (sparse(d + 1, S.kids.data() + (S.soff[s] - S.scnt[s]), S.scnt[s], j - 1, pi + 1)) return 1;
            if (stopped()) return 0;
        }
        return 0;
    }
};

// The path of a successful search as a flat in the coordinates of the root.
static void pa_lift(const PField& F, const PCtx& c, int n, int k, PFlat& fl) {
    const int D = c.ddepth;
    auto up = [&](std::vector<int> dg, int from) {  // dg: digits in the coordinates of level `from`
        for (int i = from - 1; i >= 0; --i) dg.insert(dg.begin() + c.dpiv[i], 0);
        u64 r = 0;
        for (int i = n - 1; i >= 0; --i) r = r * F.p + dg[i];
        return (u32)r;
    };
    auto unp = [&](u64 x) {
        std::vector<int> dg(n - D);
        for (int i = 0; i < n - D; ++i) dg[i] = F.digit(x, i);
        return dg;
    };
    fl.a = up(unp(c.leaf), D);
    fl.basis.clear();
    for (int i = 0; i < D; ++i) {
        std::vector<int> dg(n - i, 0);
        u32 v = c.dlow[i];
        for (int t = 0; t < c.dpiv[i]; ++t, v /= F.p) dg[t] = v % F.p;
        dg[c.dpiv[i]] = 1;
        fl.basis.push_back(up(dg, i));
    }
    for (int i = D; i < k; ++i) fl.basis.push_back(up(unp(c.pdir[i]), D));
}

// Is f == value on a + span(basis), and is the basis independent?
static bool pa_constant_on_flat(const PField& F, const uint8_t* f, int n, u32 a, const std::vector<u32>& basis, int value) {
    const int k = (int)basis.size(), p = F.p;
    std::vector<std::vector<int>> rows;  // independence by elimination
    for (u32 b : basis) {
        std::vector<int> r(n);
        for (int i = 0; i < n; ++i, b /= p) r[i] = b % p;
        for (const auto& q : rows) {
            int lead = n - 1;
            while (!q[lead]) --lead;
            const int cf = r[lead] * F.inv[q[lead]] % p;
            for (int i = 0; i < n; ++i) r[i] = ((r[i] - cf * q[i]) % p + p) % p;
        }
        bool zero = true;
        for (int i = 0; i < n; ++i) zero &= r[i] == 0;
        if (zero) return false;
        rows.push_back(r);
    }
    std::vector<u64> pb(k);
    for (int i = 0; i < k; ++i) pb[i] = F.pack(basis[i], n);
    std::vector<int> cnt(k, 0);
    u64 x = F.pack(a, n);
    for (;;) {
        if (f[F.unpack(x, n)] != value) return false;
        int i = 0;
        for (; i < k; ++i) {
            x = F.add(x, pb[i]);
            if (++cnt[i] < p) break;
            cnt[i] = 0;  // p times the vector: back where it was
        }
        if (i == k) return true;
    }
}

// Exhaustive: does {x : f(x) = cst} contain a k-flat?  rootskip: first directions
// that need not be tried (see PCtx::rootskip), as a symmetry maps them to smaller ones.
static bool pa_decide(const PField& F, const uint8_t* f, int n, int cst, int k, const Options& opt, PFlat& fl, Stats& st,
                      int mode = 0, const std::vector<u64>* rootskip = nullptr) {
    const int p = F.p;
    const size_t N = F.pw[n];
    u64 total = 0;
    for (size_t x = 0; x < N; ++x) total += f[x] == cst;
    fl.c = cst;
    fl.basis.clear();
    if (k == 0) {
        if (!total) return false;
        for (size_t x = 0; x < N; ++x)
            if (f[x] == cst) {
                fl.a = (u32)x;
                break;
            }
        return true;
    }
    if (k > n || total < F.pw[k]) return false;
    std::atomic<bool> stop(false);
    std::mutex mu;
    bool found = false;
    const int nt = std::max(1, opt.threads);
    std::vector<PCtx> ctx(nt);
    for (PCtx& c : ctx) c.init(&F, n, &opt, &stop, mode);
    auto report = [&](const PCtx& c) {  // under mu
        if (found) return;
        found = true;
        pa_lift(F, c, n, k, fl);
        stop = true;
    };
    bool sparse_root = k == 1 || mode == 1;
    if (!sparse_root && mode != 2) sparse_root = ctx[0].sparse_cost(total, n, k, 0) < ctx[0].dense_cost(n, k, 0);
    if (sparse_root) {
        std::vector<u64> L;
        L.reserve(total);
        u64 px = 0;
        for (size_t x = 0; x < N; ++x, px = F.next(px))
            if (f[x] == cst) L.push_back(px);
        PCtx& root = ctx[0];
        root.ddepth = 0;
        root.sdim = n;
        root.rootskip = rootskip;
        root.roothi = n - k;
        if (nt > 1)
            root.kid_hook = [&](PCtx::Sp& S, int j, int pi) {
                std::atomic<size_t> next(0);
                pa_parallel(nt, [&](int ti) {
                    PCtx& w = ctx[ti];
                    for (;;) {
                        const size_t i = next.fetch_add(1);
                        if (i >= S.viable.size() || stop.load()) break;
                        const u32 s = S.viable[i];
                        w.ddepth = 0;
                        w.sdim = n;
                        w.pdir[0] = S.sdir[s];
                        // root's own scratch (sp[0]) holds the lists, so the workers start one level down
                        if (w.sparse(1, S.kids.data() + (S.soff[s] - S.scnt[s]), S.scnt[s], j - 1, pi + 1)) {
                            std::lock_guard<std::mutex> g(mu);
                            report(w);
                        }
                    }
                });
                return found;
            };
        const bool hit = root.sparse(0, L.data(), L.size(), k, 0);
        if (hit && !found) report(root);
    } else {
        const int c = F.cdim(k), mc = n - c, hi = n - k;
        const size_t ncell = F.pw[mc];
        std::vector<u64> S(ncell, 0);
        for (size_t x = 0; x < N; ++x)
            if (f[x] == cst) S[x % ncell] |= 1ULL << (x / ncell);
        const u64 ncand = (F.pw[hi + 1] - 1) / (p - 1), need = F.pw[k - 1];
        ++st.dense_nodes;
        std::atomic<u64> next(0);
        pa_parallel((int)std::min<u64>(nt, ncand), [&](int ti) {
            PCtx& w = ctx[ti];
            std::vector<u64>& ob = w.tmp[1];
            ob.resize(F.pw[mc - 1]);
            for (;;) {
                u64 idx = next.fetch_add(1);
                if (idx >= ncand || stop.load()) break;
                // Alternately from both ends: most flats have the largest pivot, but with
                // symmetry the flats that are left tend to have a small first direction.
                idx = idx & 1 ? ncand - 1 - idx / 2 : idx / 2;
                int pi = hi;  // the largest pivot first
                while (idx >= F.pw[pi]) idx -= F.pw[pi--];  // idx is now the low part of the direction
                if (rootskip) {
                    const u64 v = idx + F.pw[pi];
                    if (((*rootskip)[v >> 6] >> (v & 63)) & 1) {
                        ++w.st.sym_skipped;
                        continue;
                    }
                }
                const size_t P = F.pw[pi];
                if (w.trv[0].size() < (size_t)(p - 1) * P) w.trv[0].resize((size_t)(p - 1) * P);
                ++w.st.cands;
                w.st.work += (u64)p * F.pw[mc - 1];
                w.build_tr(w.trv[0].data(), pi, (u32)idx);
                if (pa_quot(p, S.data(), ob.data(), P, F.pw[mc - 1 - pi], w.trv[0].data(), need, F.pw[c]) < need) continue;
                w.dpiv[0] = pi;
                w.dlow[0] = (u32)idx;
                if (w.enter(1, ob.data(), n - 1, k - 1, pi, c)) {
                    std::lock_guard<std::mutex> g(mu);
                    report(w);
                }
            }
        });
    }
    for (PCtx& c : ctx) st.add(c.st);
    if (found && !pa_constant_on_flat(F, f, n, fl.a, fl.basis, cst)) {
        fprintf(stderr, "internal error: the search returned a flat on which f is not constant\n");
        abort();
    }
    return found;
}

// A maximal flat inside the set L (packed points, sorted), by a randomised
// greedy descent: among a few sampled directions of lines in the set, take
// one that keeps many lines.  Returns its dimension.
static int pa_greedy(const PField& F, std::vector<u64> T, std::mt19937_64& rng, std::vector<u64>& dirs, u64& leaf) {
    const int p = F.p;
    PHash H;
    dirs.clear();
    auto line = [&](u64 z, u64 dv) {
        for (int t = 1; t < p; ++t) {
            z = F.add(z, dv);
            if (!H.has(z)) return false;
        }
        return true;
    };
    std::vector<u64> C;
    while (T.size() >= (size_t)p) {
        const size_t N = T.size();
        H.build(T.data(), N);
        u64 best = 0;
        size_t bests = 0;
        int seen = 0;
        for (int tries = 0; tries < 6000 && seen < 12; ++tries) {
            const u64 x = T[rng() % N], y = T[rng() % N];
            if (x == y) continue;
            u64 dv = F.sub(y, x);
            dv = F.scale(dv, F.inv[F.digit(dv, F.lead(dv))]);
            if (!line(x, dv)) continue;
            ++seen;
            const size_t step = std::max<size_t>(1, N / 512);
            size_t sc = 0;
            for (size_t i = rng() % step; i < N; i += step) sc += line(T[i], dv);
            if (!best || sc > bests) best = dv, bests = sc;
        }
        if (!best) break;
        const int pi = F.lead(best);
        C.clear();
        for (u64 z : T)
            if (F.digit(z, pi) == 0 && line(z, best)) C.push_back(z);
        if (C.empty()) break;
        dirs.push_back(best);
        T.swap(C);
    }
    leaf = T[0];
    return (int)dirs.size();
}

// The vectors v with f(x + v) = f(x) for all x, as a reduced basis:
// leading digits 1, increasing leading coordinates.
static std::vector<u32> pa_linear_structures(const PField& F, const uint8_t* f, int n) {
    const int p = F.p;
    const size_t N = F.pw[n];
    std::vector<u32> basis;
    std::vector<int> piv;
    std::mt19937_64 rng(12345);
    for (int pi = 0; pi < n; ++pi)
        for (u32 v = (u32)F.pw[pi]; v < 2 * F.pw[pi]; ++v) {
            if (f[v] != f[0]) continue;
            bool red = true;
            for (int q : piv) red &= (v / F.pw[q]) % p == 0;
            if (!red) continue;
            bool ok = true;
            for (int t = 0; t < 8 && ok; ++t) {
                const u32 x = (u32)(rng() % N);
                ok = f[pa_addidx(p, n, x, v)] == f[x];
            }
            if (!ok) continue;
            const u64 pv = F.pack(v, n);
            u64 px = 0;
            for (size_t x = 0; x < N && ok; ++x, px = F.next(px)) ok = f[F.unpack(F.add(px, pv), n)] == f[x];
            if (ok) {
                basis.push_back(v);
                piv.push_back(pi);
                break;  // one basis vector per leading coordinate
            }
        }
    return basis;
}

#include "psym.hpp"

struct PResult {
    int p = 0, n = 0, k = -1, value = 0;
    u32 a = 0;
    std::vector<u32> basis;   // f == value on a + span(basis)
    int lin_dim = 0;          // invariant linear structures factored out
    std::vector<int> ub;      // spectral upper bound per constant
    int greedy_k = -1;
    int searches = 0;
    double max_walsh = 0;
    bool bent = false;
    // symmetry
    bool sym_looked = false;  // an exhaustive search was needed, so the maps were looked at
    int sym_offered = 0;      // maps offered
    int sym_used = 0;         // maps with f(A x) = sigma(f(x))
    int sym_classes = 0;      // classes of constants with points under their sigmas (one constant per class is searched)
    double sym_order = 0;     // size of the largest group used to skip first directions (as far as it was listed)
    std::string sym_note;
    // weak normality (weak.hpp): k is then the largest dimension of a flat on which f is affine,
    // f(a + sum t_i basis[i]) = value + sum t_i slope[i]
    bool weak = false;
    std::vector<int> slope;
    int const_k = -1;         // dimension of the largest flat found on which f is constant
    int walsh_cap = -1;       // largest k with p^k <= max |Walsh coefficient of f|
    int graph_dim = -1;       // dimension of the flat that was looked for in the graph of f; -1: no search there
    bool graph_found = false;
    bool sym_graph = false;   // the symmetry fields describe the search in the graph
    Stats st;
    double seconds = 0;
};

// mode: see PCtx::mode (for the tests).  target >= 0: only decide whether f is target-normal.
// maps: candidate symmetries (psym.hpp); each is checked before it is used.
static PResult pa_solve(const PT& f0, int p, int n, const Options& opt, int target = -1, int mode = 0, int restarts = 3,
                        const std::vector<PMap>* maps = nullptr) {
    const double t0 = now_sec();
    PResult R;
    R.p = p;
    R.n = n;
    PField F;
    F.init(p);
    // 1. invariant linear structures: f(x) = g(x with the pivot coordinates deleted)
    const std::vector<u32> ls = pa_linear_structures(F, f0.data(), n);
    const int r = (int)ls.size(), m = n - r;
    R.lin_dim = r;
    std::vector<int> free_pos;  // the coordinates that remain
    {
        std::vector<char> isp(n, 0);
        for (u32 b : ls) {
            int lead = 0;
            for (u32 t = b; t >= (u32)p; t /= p) ++lead;
            isp[lead] = 1;
        }
        for (int i = 0; i < n; ++i)
            if (!isp[i]) free_pos.push_back(i);
    }
    auto widen = [&](u32 y) {  // a point of F_p^m as a point of F_p^n
        u32 x = 0;
        for (int i = 0; i < m; ++i, y /= p) x += (u32)(y % p * F.pw[free_pos[i]]);
        return x;
    };
    PT g;
    if (r) {
        g.resize(F.pw[m]);
        for (size_t y = 0; y < g.size(); ++y) g[y] = f0[widen((u32)y)];
    }
    const PT& f = r ? g : f0;
    const size_t N = F.pw[m];
    // 2. spectral bounds
    const PSpec sp = pa_spectrum(F, f.data(), m, opt.threads);
    R.max_walsh = sp.max_walsh * (double)F.pw[r];
    R.bent = sp.bent && r == 0;
    R.ub.assign(p, -1);
    int top = 0;
    for (int c = 0; c < p; ++c) {
        if (sp.ub[c] >= 0) R.ub[c] = sp.ub[c] + r;
        top = std::max(top, sp.ub[c]);
    }
    if (opt.verbose) {
        fprintf(stderr, "p = %d, n = %d (%d after removing linear structures), max |Walsh| = %.3f, bounds:", p, n, m,
                R.max_walsh);
        for (int c = 0; c < p; ++c) fprintf(stderr, " %d", R.ub[c]);
        fprintf(stderr, "\n");
    }
    int c_one = -1;  // Options::one_const: the only constant that is looked at
    if (opt.one_const)
        for (c_one = 0; c_one < p - 1 && sp.size[c_one] == 0;) ++c_one;
    // 3. greedy lower bound
    PFlat best;
    best.c = f[0];
    best.a = 0;
    int bk = 0;
    {
        std::mt19937_64 rng(2024);
        std::vector<u64> L, dirs;
        const int rounds = N > 2000000 ? 1 : std::max(1, restarts);
        for (int c = 0; c < p && bk < top; ++c) {
            if (sp.ub[c] <= bk || (c_one >= 0 && c != c_one)) continue;
            L.clear();
            u64 px = 0;
            for (size_t x = 0; x < N; ++x, px = F.next(px))
                if (f[x] == c) L.push_back(px);
            for (int it = 0; it < rounds && bk < sp.ub[c]; ++it) {
                u64 leaf;
                const int kk = pa_greedy(F, L, rng, dirs, leaf);
                if (kk <= bk) continue;
                PFlat fl;
                fl.c = c;
                fl.a = (u32)F.unpack(leaf, m);
                for (u64 dv : dirs) fl.basis.push_back((u32)F.unpack(dv, m));
                if (pa_constant_on_flat(F, f.data(), m, fl.a, fl.basis, c)) best = fl, bk = kk;
            }
        }
    }
    R.greedy_k = bk + r;
    if (opt.verbose) fprintf(stderr, "greedy: flat of dimension %d\n", bk + r);
    // 4. exhaustive searches from the top
    // Symmetry, prepared when the first exhaustive search is needed.
    std::vector<PMat> symM;                // linear parts of the maps that passed the check
    std::vector<std::vector<int>> symS;    // and how they permute the constants
    std::vector<int> crep(p);              // smallest constant of the class of c
    std::vector<PStab> stab(p);
    bool sym_ready = false;
    std::mt19937_64 srng(77);
    auto prepare_sym = [&]() {
        sym_ready = true;
        for (int c = 0; c < p; ++c) crep[c] = c;
        if (!maps || maps->empty()) return;
        R.sym_looked = true;
        R.sym_offered = (int)maps->size();
        if (r) {
            R.sym_note = "not used: the function has invariant linear structures";
            return;
        }
        for (const PMap& a : *maps) {
            PMat M;
            std::vector<int> sg;
            if (!pa_map_check(F, f.data(), m, a, M, sg)) continue;
            symM.push_back(M);
            symS.push_back(sg);
            for (int c = 0; c < p; ++c) {  // c and sg[c] are in one class
                const int x = crep[c], y = crep[sg[c]];
                if (x != y)
                    for (int& q : crep)
                        if (q == std::max(x, y)) q = std::min(x, y);
            }
        }
        R.sym_used = (int)symM.size();
        for (int c = 0; c < p; ++c) R.sym_classes += crep[c] == c && sp.size[c] > 0;
        if (opt.verbose)
            fprintf(stderr, "symmetry: %d of %d maps are symmetries; %d classes of constants\n", R.sym_used, R.sym_offered,
                    R.sym_classes);
    };
    auto search = [&](int t) {
        if (!sym_ready) prepare_sym();
        for (int c = 0; c < p; ++c) {
            if (sp.ub[c] < t || crep[c] != c || (c_one >= 0 && c != c_one)) continue;
            ++R.searches;
            const double ts = now_sec();
            PFlat fl;
            bool hit;
            if (!symM.empty() && !stab[c].ready) pa_stabiliser(F, c, symM, symS, stab[c]);
            if (!symM.empty() && !stab[c].elems.empty() && t >= 2) {
                // Coordinates y, x = Bm y, in which the first directions tried span an
                // invariant subspace, as far as there is one of that dimension.  A subgroup can
                // have one where the whole group has not, so the subspaces of the group and of
                // each generator alone are tried; the directions skipped are always those that
                // some element of the whole group makes smaller.
                const int hi = m - t;
                PMat Bm, Bi;
                std::vector<u64> skip;
                u64 nskip = 0;
                int wdim = -1;
                const size_t ngen = stab[c].gens.size();
                for (size_t o = 0; o <= (ngen > 1 ? ngen : 0); ++o) {
                    const std::vector<PMat> sub = o ? std::vector<PMat>(1, stab[c].gens[o - 1]) : stab[c].gens;
                    const PSub W = pa_invariant_subspace(F, m, sub, hi + 1, srng);
                    PSub all = W;
                    PMat B2(m), B2i;
                    int col = 0;
                    for (const PVec& v : W.b) {
                        for (int q = 0; q < m; ++q) B2.at(q, col) = v[q];
                        ++col;
                    }
                    for (int i = 0; i < m; ++i) {
                        PVec e(m, 0);
                        e[i] = 1;
                        if (all.add(e, F)) B2.at(i, col++) = 1;
                    }
                    pm_inv(B2, F, B2i);
                    std::vector<PMat> el;
                    for (const PMat& e : stab[c].elems) el.push_back(pm_mul(B2i, pm_mul(e, B2, p), p));
                    u64 ns = 0;
                    std::vector<u64> sk = pa_root_skip(F, hi, el, ns);
                    if (wdim < 0 || ns > nskip) {
                        Bm = B2, Bi = B2i, nskip = ns, wdim = W.dim();
                        skip.swap(sk);
                    }
                }
                PT g2(N);
                pa_images(F, m, Bm, 0, [&](u32 y, u64 x) {
                    g2[y] = f[F.unpack(x, m)];
                    return true;
                });
                R.sym_order = std::max(R.sym_order, (double)stab[c].elems.size() + 1);
                if (opt.verbose)
                    fprintf(stderr, "  symmetry: %zu maps of the set to itself%s, invariant subspace of dimension %d (wanted %d), "
                                    "%llu of %llu first directions skipped\n",
                            stab[c].elems.size() + 1, stab[c].complete ? "" : " listed", wdim, hi + 1,
                            (unsigned long long)nskip, (unsigned long long)((F.pw[hi + 1] - 1) / (p - 1)));
                hit = pa_decide(F, g2.data(), m, c, t, opt, fl, R.st, mode, &skip);
                if (hit) {  // back to the coordinates x
                    fl.a = pv_index(pm_apply(Bm, pv_digits(fl.a, m, p), p), p);
                    for (u32& bv : fl.basis) bv = pv_index(pm_apply(Bm, pv_digits(bv, m, p), p), p);
                    if (!pa_constant_on_flat(F, f.data(), m, fl.a, fl.basis, c)) {
                        fprintf(stderr, "internal error: the flat found with symmetry is not one of f\n");
                        abort();
                    }
                }
            } else hit = pa_decide(F, f.data(), m, c, t, opt, fl, R.st, mode);
            if (opt.verbose)
                fprintf(stderr, "  %d-flat with f = %d: %s (%.2f s)\n", t + r, c, hit ? "found" : "none", now_sec() - ts);
            if (hit) {
                best = fl;
                bk = t;
                return true;
            }
        }
        return false;
    };
    if (target >= 0) {
        if (bk < target - r && target - r <= top) search(target - r);
    } else {
        for (int t = top; t > bk; --t)
            if (search(t)) break;
    }
    // 5. back to the coordinates of f0
    R.k = bk + r;
    R.value = best.c;
    R.a = widen(best.a);
    for (u32 b : best.basis) R.basis.push_back(widen(b));
    for (u32 b : ls) R.basis.push_back(b);
    if (!pa_constant_on_flat(F, f0.data(), n, R.a, R.basis, R.value)) {
        fprintf(stderr, "internal error: f is not constant on the flat found\n");
        abort();
    }
    R.seconds = now_sec() - t0;
    return R;
}
