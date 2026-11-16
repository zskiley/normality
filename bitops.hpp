// Bit-level primitives for the normality solver.
//
// A subset T of F_2^m is stored as a bitset of 2^m bits: bit x of the array is
// set iff x is in T (x read as an m-bit integer, coordinate i = bit i of x).
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#if defined(__BMI2__) || defined(__AVX2__)
#include <immintrin.h>
#endif

typedef uint64_t u64;
typedef uint32_t u32;

// MLO[b] = positions whose index has bit b clear.
static const u64 MLO[6] = {0x5555555555555555ULL, 0x3333333333333333ULL,
                           0x0F0F0F0F0F0F0F0FULL, 0x00FF00FF00FF00FFULL,
                           0x0000FFFF0000FFFFULL, 0x00000000FFFFFFFFULL};
// BLK[b] = positions that are multiples of 2^b (lowest bit of every 2^b-block).
static const u64 BLK[7] = {0xFFFFFFFFFFFFFFFFULL, 0x5555555555555555ULL,
                           0x1111111111111111ULL, 0x0101010101010101ULL,
                           0x0001000100010001ULL, 0x0000000100000001ULL, 1ULL};

static inline int popc(u64 x) { return __builtin_popcountll(x); }
static inline int ctz64(u64 x) { return __builtin_ctzll(x); }
static inline int msb32(u32 x) { return 31 - __builtin_clz(x); }
static inline size_t nwords(int m) { return m <= 6 ? 1 : (size_t)1 << (m - 6); }

// Swap neighbouring groups of 2^b bits: result bit i = input bit i ^ 2^b.
static inline u64 swapb(u64 w, int b) {
    int s = 1 << b;
    return ((w >> s) & MLO[b]) | ((w & MLO[b]) << s);
}
// XOR-translate inside a word: result bit i = input bit (i ^ vb), vb < 64.
static inline u64 perm6(u64 w, unsigned vb) {
    while (vb) { w = swapb(w, __builtin_ctz(vb)); vb &= vb - 1; }
    return w;
}
// r has bits only at positions with index-bit p clear; delete coordinate p.
static inline u64 compress_p(u64 r, int p) {
    for (int q = p; q < 5; ++q) r = (r | (r >> (1 << q))) & MLO[q + 1];
    return r;
}
// OR every 2^b-block of the word into the block's lowest bit (other bits 0).
static inline u64 fold_blocks(u64 w, int b) {
    for (int q = 0; q < b; ++q) w |= w >> (1 << q);
    return w & BLK[b];
}
// Gather the bits at multiples of 2^b (as left by fold_blocks) into the low bits.
static inline u64 pack_blocks(u64 y, int b) {
#ifdef __BMI2__
    return _pext_u64(y, BLK[b]);
#else
    for (int q = b - 1; q >= 0; --q) y = compress_p(y, q);
    return y;
#endif
}
static inline u32 remove_bit(u32 x, int p) {
    return ((x >> (p + 1)) << p) | (x & ((1u << p) - 1));
}
static inline u32 insert_bit0(u32 y, int p) {
    return ((y >> p) << (p + 1)) | (y & ((1u << p) - 1));
}
static inline bool getbit(const u64* T, u32 x) { return (T[x >> 6] >> (x & 63)) & 1; }

static inline u64 popcount_arr(const u64* T, size_t nw) {
    u64 c = 0;
    for (size_t i = 0; i < nw; ++i) c += popc(T[i]);
    return c;
}
// First set bit at index >= from, or -1.
static inline int64_t first_set(const u64* T, size_t nw, u64 from = 0) {
    size_t i = from >> 6;
    if (i >= nw) return -1;
    u64 w = T[i] & (~0ULL << (from & 63));
    for (;;) {
        if (w) return (int64_t)(i * 64 + ctz64(w));
        if (++i >= nw) return -1;
        w = T[i];
    }
}

// Quotient step.  v != 0 has leading bit p.  The result lives in F_2^(m-1)
// (coordinate p deleted) and contains y iff both points of the coset
// {x, x^v} are in T, where x is y with a 0 inserted at position p.
// Returns the number of such cosets.
static u64 quotient(const u64* T, int m, u32 v, u64* out) {
    int p = msb32(v);
    u32 vlo = v ^ (1u << p);
    u64 cnt = 0;
    if (p >= 6) {
        size_t W = (size_t)1 << (p - 6), nblk = (size_t)1 << (m - 1 - p);
        size_t vw = vlo >> 6;
        unsigned vb = vlo & 63;
        for (size_t blk = 0; blk < nblk; ++blk) {
            const u64* A = T + blk * 2 * W;
            const u64* B = A + W;
            u64* O = out + blk * W;
            for (size_t i = 0; i < W; ++i) {
                u64 o = A[i] & perm6(B[i ^ vw], vb);
                O[i] = o;
                cnt += popc(o);
            }
        }
    } else {
        int s = 1 << p;
        u64 M = MLO[p];
        if (m <= 6) {
            u64 w = T[0];
            u64 r = compress_p((w & M) & perm6((w >> s) & M, vlo), p);
            out[0] = r;
            cnt = popc(r);
        } else {
            size_t nout = nwords(m) / 2;
            for (size_t i = 0; i < nout; ++i) {
                u64 w0 = T[2 * i], w1 = T[2 * i + 1];
                u64 r0 = compress_p((w0 & M) & perm6((w0 >> s) & M, vlo), p);
                u64 r1 = compress_p((w1 & M) & perm6((w1 >> s) & M, vlo), p);
                u64 o = r0 | (r1 << 32);
                out[i] = o;
                cnt += popc(o);
            }
        }
    }
    return cnt;
}

// out bit i = (a[i] < v) for i < n (n a multiple of 64 or the whole array); `acc`
// set: or into out instead of overwriting it.
static void below_mask(const u32* a, size_t n, u32 v, u64* out, bool acc) {
    size_t i = 0, w = 0;
#ifdef __AVX2__
    const __m256i bias = _mm256_set1_epi32((int)0x80000000u);
    const __m256i vv = _mm256_set1_epi32((int)(v ^ 0x80000000u));
    for (; i + 64 <= n; i += 64, ++w) {
        u64 bits = 0;
        for (int k = 0; k < 8; ++k) {
            __m256i x = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(a + i + 8 * k)), bias);
            bits |= (u64)(unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(vv, x))) << (8 * k);
        }
        out[w] = acc ? out[w] | bits : bits;
    }
#endif
    for (; i < n; i += 64, ++w) {
        u64 bits = 0;
        size_t lim = n - i < 64 ? n - i : 64;
        for (size_t b = 0; b < lim; ++b) bits |= (u64)(a[i + b] < v) << b;
        out[w] = acc ? out[w] | bits : bits;
    }
}

// The image of T in F_2^m / <v>: coset {x, x^v} is in the result if x or x^v is in T.
static void or_quotient(const u64* T, int m, u32 v, u64* out) {
    int p = msb32(v);
    u32 vlo = v ^ (1u << p);
    if (p >= 6) {
        size_t W = (size_t)1 << (p - 6), nblk = (size_t)1 << (m - 1 - p);
        size_t vw = vlo >> 6;
        unsigned vb = vlo & 63;
        for (size_t blk = 0; blk < nblk; ++blk) {
            const u64* A = T + blk * 2 * W;
            const u64* B = A + W;
            u64* O = out + blk * W;
            for (size_t i = 0; i < W; ++i) O[i] = A[i] | perm6(B[i ^ vw], vb);
        }
    } else {
        int s = 1 << p;
        u64 M = MLO[p];
        if (m <= 6) {
            u64 w = T[0];
            out[0] = compress_p((w & M) | perm6((w >> s) & M, vlo), p);
        } else {
            size_t nout = nwords(m) / 2;
            for (size_t i = 0; i < nout; ++i) {
                u64 w0 = T[2 * i], w1 = T[2 * i + 1];
                u64 r0 = compress_p((w0 & M) | perm6((w0 >> s) & M, vlo), p);
                u64 r1 = compress_p((w1 & M) | perm6((w1 >> s) & M, vlo), p);
                out[i] = r0 | (r1 << 32);
            }
        }
    }
}

// In-place Walsh-Hadamard transform, length 2^m.
template <class I> static void wht(I* a, int m) {
    size_t n = (size_t)1 << m;
    for (size_t h = 1; h < n; h <<= 1)
        for (size_t i = 0; i < n; i += 2 * h)
            for (size_t j = i; j < i + h; ++j) {
                I x = a[j], y = a[j + h];
                a[j] = x + y;
                a[j + h] = x - y;
            }
}

// Does X, a subset of F_2^D with D <= 6 held in one word, contain an r-flat
// whose direction space has all leading bits >= lo?  (Decision only.)
static bool flat64(u64 X, int D, int r, int lo) {
    if (r == 0) return X != 0;
    if (popc(fold_blocks(X, lo)) < (1 << r)) return false;
    if (r == 1) return true;  // two points in different 2^lo-blocks
    if (r == D - 1 && lo == 0) {
        // A hyperplane lies inside X iff the complement fits in the other half,
        // i.e. iff the complement does not affinely span the whole space.
        u64 C = ~X & (D == 6 ? ~0ULL : (1ULL << (1 << D)) - 1);
        if (!C) return true;
        u32 c0 = (u32)ctz64(C), bas[6] = {0, 0, 0, 0, 0, 0};
        int rank = 0;
        for (C &= C - 1; C; C &= C - 1) {
            u32 v = (u32)ctz64(C) ^ c0;
            while (v) {
                int h = msb32(v);
                if (!bas[h]) { bas[h] = v; ++rank; break; }
                v ^= bas[h];
            }
            if (rank == D) return false;
        }
        return true;
    }
    int pmax = D - r, need = 1 << (r - 1);
    for (int p = lo; p <= pmax; ++p) {
        u64 a = X & MLO[p], b = (X >> (1 << p)) & MLO[p];
        for (unsigned g = 0; g < (1u << p); ++g) {  // low part of v in Gray order
            if (g) b = swapb(b, __builtin_ctz(g));
            u64 q = a & b;
            if (popc(q) >= need && flat64(compress_p(q, p), D - 1, r - 1, p)) return true;
        }
    }
    return false;
}

// Projection of T (m >= 6 dims) onto its top six coordinates: bit b of the
// result is set iff the b-th block of 2^(m-6) points is non-empty.
static u64 shadow64(const u64* T, int m) {
    if (m <= 6) return T[0];
    u64 mask = 0;
#ifdef __AVX2__
    const __m256i z = _mm256_setzero_si256();
#define LD(p) _mm256_loadu_si256((const __m256i*)(p))
    if (m >= 14) {  // blocks of W >= 4 words
        size_t W = (size_t)1 << (m - 12);
        for (int b = 0; b < 64; ++b) {
            __m256i acc = z;
            for (size_t i = 0; i < W; i += 4) acc = _mm256_or_si256(acc, LD(T + b * W + i));
            mask |= (u64)!_mm256_testz_si256(acc, acc) << b;
        }
        return mask;
    }
    if (m == 13) {  // two words per block
        for (int b = 0; b < 64; b += 2) {
            int e = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(LD(T + 2 * b), z)));
            mask |= (u64)(((e & 3) != 3) | (((e & 12) != 12) << 1)) << b;
        }
        return mask;
    }
    if (m == 12) {
        for (int b = 0; b < 64; b += 4) {
            u64 e = (u64)_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(LD(T + b), z)));
            mask |= (e ^ 15) << b;
        }
        return mask;
    }
    if (m == 11) {  // 32-bit blocks
        for (int i = 0; i < 32; i += 4) {
            u64 e = (u64)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(LD(T + i), z)));
            mask |= (e ^ 0xFF) << (2 * i);
        }
        return mask;
    }
    if (m == 9) {  // 8-bit blocks
        for (int i = 0; i < 8; i += 4) {
            u32 e = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(LD(T + i), z));
            mask |= (u64)(u32)~e << (8 * i);
        }
        return mask;
    }
#undef LD
#endif
    if (m >= 12) {
        size_t W = (size_t)1 << (m - 12);
        for (int b = 0; b < 64; ++b) {
            u64 acc = 0;
            for (size_t i = 0; i < W; ++i) acc |= T[b * W + i];
            mask |= (u64)(acc != 0) << b;
        }
        return mask;
    }
    int lo = m - 6, per = 12 - m;  // a word holds 2^per blocks of 2^lo bits
    size_t nw = nwords(m);
    for (size_t i = 0; i < nw; ++i) mask |= pack_blocks(fold_blocks(T[i], lo), lo) << (i << per);
    return mask;
}

// Necessary condition for "T (m dims) contains a j-flat with leading bits
// >= lo", given mask = shadow64(T).  Projecting onto the coordinates >= t is
// injective on such a flat when t <= lo, and loses at most t - lo dimensions
// otherwise.  For m <= 6 the test is exact.
static bool top6_ok(u64 mask, int m, int j, int lo) {
    if (j == 0) return mask != 0;
    if (m <= 6) return flat64(mask, m, j, lo);
    int t = m - 6;
    if (lo >= t) return flat64(mask, 6, j, lo - t);
    int r = j - (t - lo);
    return r < 1 || flat64(mask, 6, r, 0);
}

// Shadow test with multiplicities, for lo < t = m - 6 (s = t - lo).  A j-flat
// with leading bits >= lo has i of them among the top six coordinates and
// j - i (at most s) below.  Its image in the bucket space is an i-flat, and
// inside each of those buckets it is a (j-i)-flat that meets 2^(j-i) distinct
// 2^lo-blocks.  So for some i, the buckets holding >= 2^(j-i) non-empty
// blocks must contain an i-flat.  Returns the smallest such i, or -1.
static int multi_shadow_min(const u32* cnt, int j, int s) {
    int ihi = j < 6 ? j : 6, ilo = j - s > 0 ? j - s : 0;
    for (int i = ilo; i <= ihi; ++i) {
        u32 needb = 1u << (j - i);
        u64 M = 0;
        for (int b = 0; b < 64; ++b) M |= (u64)(cnt[b] >= needb) << b;
        if (flat64(M, 6, i, 0)) return i;
    }
    return -1;
}

// Write the indices of the set bits of T (nw words) to out, ascending.
static inline size_t extract_bits(const u64* T, size_t nw, u32* out) {
    size_t n = 0, i = 0;
#ifdef __AVX2__
    const __m256i z = _mm256_setzero_si256();
    for (; i + 64 <= nw; i += 64) {  // bitmap of non-zero words first: few branch mispredictions
        u64 nz = 0;
        for (int k = 0; k < 64; k += 4)
            nz |= (u64)(15 ^ _mm256_movemask_pd(_mm256_castsi256_pd(
                      _mm256_cmpeq_epi64(_mm256_loadu_si256((const __m256i*)(T + i + k)), z)))) << k;
        while (nz) {
            size_t wi = i + ctz64(nz);
            nz &= nz - 1;
            u64 w = T[wi];
            do { out[n++] = (u32)(wi * 64 + ctz64(w)); w &= w - 1; } while (w);
        }
    }
#endif
    for (; i < nw; ++i)
        for (u64 w = T[i]; w; w &= w - 1) out[n++] = (u32)(i * 64 + ctz64(w));
    return n;
}
