// Input and a few families of functions F_p^n -> F_p for tests and examples.
#pragma once
#include "pary.hpp"
#include <string>

// Table as text.  For p <= 7 every digit character is one value (white space
// and commas are ignored); for larger p the values are numbers separated by
// white space or commas.  Returns n, or -1 on bad input.
static int pa_parse(const std::string& s, int p, PT& f) {
    f.clear();
    if (p <= 7) {
        for (char ch : s) {
            if (ch >= '0' && ch < '0' + p) f.push_back((uint8_t)(ch - '0'));
            else if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t' && ch != ',') return -1;
        }
    } else {
        long cur = -1;
        for (size_t i = 0; i <= s.size(); ++i) {
            const char ch = i < s.size() ? s[i] : ' ';
            if (ch >= '0' && ch <= '9') {
                cur = (cur < 0 ? 0 : cur) * 10 + (ch - '0');
                if (cur >= p) return -1;
            } else if (ch == ' ' || ch == '\n' || ch == '\r' || ch == '\t' || ch == ',') {
                if (cur >= 0) f.push_back((uint8_t)cur);
                cur = -1;
            } else return -1;
        }
    }
    int n = 0;
    u64 q = 1;
    while (q < f.size()) q *= p, ++n;
    if (f.empty() || q != f.size()) return -1;
    return n;
}

static PT pa_random(int p, int n, u64 seed) {
    PField F;
    F.init(p);
    std::mt19937_64 rng(seed);
    PT f(F.pw[n]);
    for (auto& v : f) v = (uint8_t)(rng() % p);
    return f;
}

// GF(p^n) = F_p[x] / (m(x)) with x a primitive element; the element
// c_0 + c_1 x + ... is the point with digits c_0, c_1, ...
struct PGF {
    int p = 0, n = 0;
    u32 q = 0;
    std::vector<u32> ex;      // ex[i] = x^i as a point, 0 <= i < q - 1
    std::vector<uint8_t> tr;  // tr[i] = Tr(x^i)

    typedef std::vector<int> Poly;
    static Poly mulmod(const Poly& a, const Poly& b, const Poly& red, int p) {  // x^n = sum red[i] x^i
        const int n = (int)red.size();
        std::vector<int> t(2 * n - 1, 0);
        for (int i = 0; i < n; ++i)
            if (a[i])
                for (int j = 0; j < n; ++j) t[i + j] = (t[i + j] + a[i] * b[j]) % p;
        for (int i = 2 * n - 2; i >= n; --i)
            if (t[i])
                for (int j = 0; j < n; ++j) t[i - n + j] = (t[i - n + j] + t[i] * red[j]) % p;
        t.resize(n);
        return t;
    }
    static Poly powmod(Poly a, u64 e, const Poly& red, int p) {
        Poly r(red.size(), 0);
        r[0] = 1;
        for (; e; e >>= 1, a = mulmod(a, a, red, p))
            if (e & 1) r = mulmod(r, a, red, p);
        return r;
    }
    bool init(int prime, int dim) {
        PField F;
        if (!F.init(prime) || dim < 1 || F.pw[dim] > 60000000) return false;
        p = prime;
        n = dim;
        q = (u32)F.pw[n];
        std::vector<u64> fac;
        for (u64 t = q - 1, d = 2; t > 1; ++d) {
            if (d * d > t) d = t;
            if (t % d == 0) {
                fac.push_back(d);
                while (t % d == 0) t /= d;
            }
        }
        Poly red(n), one(n, 0), xx(n, 0);
        one[0] = 1;
        if (n == 1) {  // the field is F_p itself; "x" is a primitive root g, i.e. m(x) = x - g
            for (int gg = 1; gg < p; ++gg) {
                bool ok = true;
                int e = 1;
                for (int i = 1; i < p - 1 && ok; ++i) {
                    e = e * gg % p;
                    ok = e != 1;
                }
                if (ok && (p == 2 || gg != 1)) {
                    red[0] = gg;
                    break;
                }
            }
        } else {
            xx[1] = 1;
            bool got = false;
            for (u64 c = 1; c < q && !got; ++c) {
                u64 t = c;
                for (int i = 0; i < n; ++i, t /= p) red[i] = (int)(t % p);
                if (!red[0]) continue;
                got = powmod(xx, q - 1, red, p) == one;
                for (size_t i = 0; i < fac.size() && got; ++i) got = powmod(xx, (q - 1) / fac[i], red, p) != one;
            }
            if (!got) return false;
        }
        ex.resize(q - 1);
        std::vector<int> e(n, 0);
        e[0] = 1;
        for (u32 i = 0; i + 1 < q; ++i) {
            u64 idx = 0;
            for (int j = n - 1; j >= 0; --j) idx = idx * p + e[j];
            ex[i] = (u32)idx;
            const int topc = e[n - 1];  // multiply by x
            for (int j = n - 1; j > 0; --j) e[j] = (e[j - 1] + topc * red[j]) % p;
            e[0] = topc * red[0] % p;
        }
        tr.resize(q - 1);
        for (u32 i = 0; i + 1 < q; ++i) {
            int s = 0;
            u64 j = i;
            for (int t = 0; t < n; ++t, j = j * p % (q - 1)) s += ex[j] % p;  // constant coefficient of x^(i p^t)
            tr[i] = (uint8_t)(s % p);
        }
        return true;
    }
};

// f(x) += Tr(g^e x^d) over GF(p^n), g the primitive element of PGF.
static bool pa_add_monomial(PT& f, int p, int n, u64 d, u64 e) {
    PGF K;
    if (!K.init(p, n)) return false;
    if (f.empty()) f.assign(K.q, 0);
    const u64 o = K.q - 1;
    d %= o;
    e %= o;
    for (u64 i = 0; i < o; ++i) {
        const u32 x = K.ex[i];
        f[x] = (uint8_t)((f[x] + K.tr[(e + i * d) % o]) % p);
    }
    return true;
}

// The maps x -> g^e x^(p^i) of GF(p^n) under which f is multiplied by a
// constant, found by trying them (on a few points first): a generator of
// those with i = 0, and one with the smallest i > 0 that has one.  These two
// generate all of them.
static void pa_field_maps(const PGF& K, const PT& f, std::vector<PMap>& out) {
    const int p = K.p, n = K.n;
    const u64 o = K.q - 1;
    std::vector<uint8_t> fx(o);
    for (u64 t = 0; t < o; ++t) fx[t] = f[K.ex[t]];
    int inv[64] = {0};
    for (int a = 1; a < p; ++a)
        for (int b = 1; b < p; ++b)
            if (a * b % p == 1) inv[a] = b;
    std::mt19937_64 rng(4242);
    std::vector<u64> ts;  // sample points with f != 0
    for (int tries = 0; tries < 4000 && ts.size() < 16; ++tries) {
        const u64 t = rng() % o;
        if (fx[t]) ts.push_back(t);
    }
    if (ts.empty()) return;
    std::vector<u64> tp(ts.size());
    u64 pwi = 1;  // p^i modulo q - 1
    for (int i = 0; i < n; ++i, pwi = pwi * p % o) {
        for (size_t s = 0; s < ts.size(); ++s) tp[s] = ts[s] * pwi % o;
        for (u64 e = i ? 0 : 1; e < o; ++e) {
            u64 img = e + tp[0];
            if (img >= o) img -= o;
            if (!fx[img]) continue;
            const int mu = fx[img] * inv[fx[ts[0]]] % p;
            bool ok = mu * f[0] % p == f[0];
            for (size_t s = 1; s < ts.size() && ok; ++s) {
                img = e + tp[s];
                if (img >= o) img -= o;
                ok = fx[img] == mu * fx[ts[s]] % p;
            }
            if (!ok) continue;
            img = e;
            for (u64 t = 0; t < o && ok; ++t) {
                ok = fx[img] == mu * fx[t] % p;
                img += pwi;
                if (img >= o) img -= o;
            }
            if (!ok) continue;
            PMap a;
            a.col.resize(n);
            for (int j = 0; j < n; ++j) a.col[j] = K.ex[(e + (u64)j * pwi) % o];
            out.push_back(a);
            if (i) return;  // together with the one for i = 0 it generates the others
            break;
        }
    }
}

// Random invertible matrix (columns as packed vectors) and shift.
static void pa_random_affine(const PField& F, int n, u64 seed, std::vector<u64>& col, u64& shift) {
    std::mt19937_64 rng(seed);
    const int p = F.p;
    for (;;) {
        std::vector<std::vector<int>> a(n, std::vector<int>(n));
        for (auto& r : a)
            for (int& v : r) v = (int)(rng() % p);
        std::vector<std::vector<int>> b = a;
        int rank = 0;
        for (int c = 0; c < n; ++c) {
            int piv = -1;
            for (int r = rank; r < n; ++r)
                if (b[r][c]) { piv = r; break; }
            if (piv < 0) break;
            std::swap(b[rank], b[piv]);
            for (int r = rank + 1; r < n; ++r) {
                const int cf = b[r][c] * F.inv[b[rank][c]] % p;
                for (int k = 0; k < n; ++k) b[r][k] = ((b[r][k] - cf * b[rank][k]) % p + p) % p;
            }
            ++rank;
        }
        if (rank < n) continue;
        col.assign(n, 0);
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) col[c] |= (u64)a[r][c] << (r * F.B);
        shift = 0;
        for (int r = 0; r < n; ++r) shift |= (u64)(rng() % p) << (r * F.B);
        return;
    }
}

// g(x) = f(A x + b) for a random invertible A and a random b.
static PT pa_scramble(const PT& f, int p, int n, u64 seed) {
    PField F;
    F.init(p);
    std::vector<u64> col;
    u64 shift;
    pa_random_affine(F, n, seed, col, shift);
    PT g(f.size());
    for (size_t x = 0; x < f.size(); ++x) {
        u64 y = shift;
        size_t t = x;
        for (int i = 0; i < n; ++i, t /= p) y = F.add(y, F.scale(col[i], (int)(t % p)));
        g[x] = f[F.unpack(y, n)];
    }
    return g;
}

// Symmetries of f as symmetries of pa_scramble(f, p, n, seed).
static void pa_scramble_maps(int p, int n, u64 seed, std::vector<PMap>& maps) {
    PField F;
    F.init(p);
    std::vector<u64> col;
    u64 shift;
    pa_random_affine(F, n, seed, col, shift);
    PMat A(n), Ai;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) A.at(r, c) = F.digit(col[c], r);
    pm_inv(A, F, Ai);
    PVec b(n);
    for (int r = 0; r < n; ++r) b[r] = F.digit(shift, r);
    for (PMap& a : maps) {  // x -> A^-1 (M (A x + b) + t - b)
        const PMat M = pm_from_cols(a.col, n, p), C = pm_mul(Ai, pm_mul(M, A, p), p);
        PVec z = pm_apply(M, b, p);
        const PVec t = pv_digits(a.t, n, p);
        for (int r = 0; r < n; ++r) z[r] = ((z[r] + t[r] - b[r]) % p + p) % p;
        a.t = pv_index(pm_apply(Ai, z, p), p);
        for (int c = 0; c < n; ++c) {
            PVec e(n);
            for (int r = 0; r < n; ++r) e[r] = C.at(r, c);
            a.col[c] = pv_index(e, p);
        }
    }
}

// Maiorana-McFarland: f(x, y) = <x, pi(y)> + g(y), x = the low n/2 coordinates,
// pi a random permutation of F_p^(n/2), g random.  Bent, and constant on the
// flat y = pi^(-1)(0).
static PT pa_mm(int p, int n, u64 seed) {
    PField F;
    F.init(p);
    const int h = n / 2;
    const size_t Q = F.pw[h];
    std::mt19937_64 rng(seed);
    std::vector<u32> perm(Q);
    for (size_t i = 0; i < Q; ++i) perm[i] = (u32)i;
    for (size_t i = Q - 1; i > 0; --i) std::swap(perm[i], perm[rng() % (i + 1)]);
    PT f(Q * Q);
    for (size_t y = 0; y < Q; ++y) {
        const int gy = (int)(rng() % p);
        for (size_t x = 0; x < Q; ++x) {
            int s = gy;
            size_t a = x, b = perm[y];
            for (int i = 0; i < h; ++i, a /= p, b /= p) s += (int)(a % p) * (int)(b % p);
            f[x + y * Q] = (uint8_t)(s % p);
        }
    }
    return f;
}

// Quadratic forms.  type 0: x0 x1 + x2 x3 + ... (plus x_(n-1)^2 if n is odd);
// type 1 (n even): x0^2 - nu x1^2 + x2 x3 + ..., nu a non-square (elliptic).
// The largest flat on which the form is constant has the dimension of a
// maximal totally isotropic subspace: n/2, n/2 - 1 and (n-1)/2.
static PT pa_quadratic(int p, int n, int type) {
    PField F;
    F.init(p);
    int nu = 2;
    for (;; ++nu) {
        bool sq = false;
        for (int a = 1; a < p; ++a) sq |= a * a % p == nu;
        if (!sq) break;
    }
    PT f(F.pw[n]);
    for (size_t x = 0; x < f.size(); ++x) {
        int dg[48] = {0};
        size_t t = x;
        for (int i = 0; i < n; ++i, t /= p) dg[i] = (int)(t % p);
        int s = 0, i = 0;
        if (type == 1) {
            s = dg[0] * dg[0] + (p - nu) * dg[1] % p * dg[1];
            i = 2;
        }
        for (; i + 1 < n; i += 2) s += dg[i] * dg[i + 1];
        if (i < n) s += dg[i] * dg[i];
        f[x] = (uint8_t)(s % p);
    }
    return f;
}

// f with the value c forced on a random k-flat.
static PT pa_plant(PT f, int p, int n, int k, int c, u64 seed) {
    PField F;
    F.init(p);
    std::vector<u64> col;
    u64 shift;
    pa_random_affine(F, n, seed, col, shift);
    std::vector<int> cnt(k, 0);
    u64 x = shift;
    for (;;) {
        f[F.unpack(x, n)] = (uint8_t)c;
        int i = 0;
        for (; i < k; ++i) {
            x = F.add(x, col[i]);
            if (++cnt[i] < p) break;
            cnt[i] = 0;
        }
        if (i == k) break;
    }
    return f;
}
