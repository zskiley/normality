// Driver: largest k such that f is constant on some k-dimensional flat.
//
//   1. quotient out invariant linear structures (f(x+v) = f(x) for all x);
//   2. upper bounds from the Walsh spectrum, one per constant c;
//   3. lower bounds: a greedy autocorrelation descent, then a budgeted search
//      restricted to flats through the origin (linear subspaces);
//   4. exhaustive decision searches for t = upper bound, upper bound - 1, ...
//      until one succeeds or t reaches the lower bound.  If automorphisms of
//      f are available (supplied candidates, or a group computed by an
//      external library), they are verified and used to search only one
//      direction space per orbit, and only one constant if some automorphism
//      maps f to f + 1.
//
// Every flat reported is re-checked point by point against the truth table.
#pragma once
#include "search.hpp"
#include <chrono>
#include <cstdio>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

struct Path {
    std::vector<u32> v;
    std::vector<int> p;
};

// A flat a + span(basis) on which f == c.
struct Flat {
    int c = 0;
    u32 a = 0;
    std::vector<u32> basis;
    int dim() const { return (int)basis.size(); }
};

struct Result {
    int n = 0, k = -1, value = 0;
    u32 a = 0;
    std::vector<u32> basis;   // f == value on a + span(basis)
    int lin_dim = 0;          // invariant linear structures factored out
    int ub[2] = {-1, -1};     // spectral upper bound for constant 0 / 1
    int greedy_k = -1;
    int searches = 0;         // exhaustive decision searches run
    int anchored_k = -1;      // set if the subspace pass improved the lower bound
    // symmetry
    bool sym_loaded = false;  // an exhaustive search was needed, so the automorphisms were looked at
    int sym_offered = 0;      // candidate maps offered (supplied and from the library)
    int sym_gens = 0;         // generators in use: linear parts of maps with f(Ax) = f(x)
    double sym_order = 0;     // order of the group they generate, 0 if it was too large to list
    bool sym_flip = false;    // some map takes f to f + 1, so both constants behave alike
    double sym_seconds = 0;   // time spent obtaining and preparing the group
    std::string sym_note;
    Stats st;
    double seconds = 0;
};

static double now_sec() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Path (directions in the coordinates of their own level) -> explicit flat.
static void lift(const Path& path, u32 leaf, Flat& fl) {
    int k = (int)path.v.size();
    fl.a = leaf;
    for (int d = k - 1; d >= 0; --d) fl.a = insert_bit0(fl.a, path.p[d]);
    fl.basis.clear();
    for (int i = 0; i < k; ++i) {
        u32 b = path.v[i];
        for (int d = i - 1; d >= 0; --d) b = insert_bit0(b, path.p[d]);
        fl.basis.push_back(b);
    }
}

static bool constant_on_flat(const u64* F, u32 a, const std::vector<u32>& basis, int value) {
    u32 x = a;
    if ((int)getbit(F, x) != value) return false;
    u64 total = (u64)1 << basis.size();
    for (u64 g = 1; g < total; ++g) {
        x ^= basis[ctz64(g)];
        if ((int)getbit(F, x) != value) return false;
    }
    return true;
}

// Keep adding the direction v that keeps the most cosets alive.  T (m dims)
// already is the quotient by the directions in `path`.  Returns the leaf point.
static u32 greedy_extend(std::vector<u64> T, int m, Path& path, std::mt19937_64* rng) {
    std::vector<int64_t> ac;
    std::vector<u64> out;
    while (m >= 1) {
        size_t N = (size_t)1 << m;
        if (popcount_arr(T.data(), nwords(m)) < 2) break;
        u32 v = 0;
        if (m <= 24) {
            ac.assign(N, 0);
            for (size_t x = 0; x < N; ++x) ac[x] = getbit(T.data(), (u32)x);
            wht(ac.data(), m);
            for (size_t x = 0; x < N; ++x) ac[x] *= ac[x];
            wht(ac.data(), m);  // ac[v] = 2^m * #{x : x, x+v in T}
            int64_t best = 0;
            for (size_t x = 1; x < N; ++x)
                if (ac[x] > best) { best = ac[x]; v = (u32)x; }
            if (best == 0) break;
            if (rng) {  // random choice among near-best directions
                int64_t cut = best - best / 16;
                u64 seen = 0;
                for (size_t x = 1; x < N; ++x)
                    if (ac[x] >= cut && (*rng)() % ++seen == 0) v = (u32)x;
            }
        } else {
            int64_t x0 = first_set(T.data(), nwords(m));
            int64_t x1 = first_set(T.data(), nwords(m), (u64)x0 + 1);
            v = (u32)x0 ^ (u32)x1;
        }
        out.assign(nwords(m - 1), 0);
        quotient(T.data(), m, v, out.data());
        T.swap(out);
        path.v.push_back(v);
        path.p.push_back(msb32(v));
        --m;
    }
    return (u32)first_set(T.data(), nwords(m));
}

// Replay a path on S (n dims), then extend it greedily to a maximal flat.
static u32 replay_extend(const u64* S, int n, Path& path) {
    std::vector<u64> T(S, S + nwords(n)), out;
    int m = n;
    for (size_t i = 0; i < path.v.size(); ++i, --m) {
        out.assign(nwords(m - 1), 0);
        quotient(T.data(), m, path.v[i], out.data());
        T.swap(out);
    }
    return greedy_extend(T, m, path, nullptr);
}

// Exhaustive: does S (n dims) contain a k-flat?  Root candidates are spread
// over the worker threads.
//   anchored    only flats through the origin are considered;
//   work_limit  non-zero turns the search into a heuristic: it may stop early,
//               and then *exhausted is set and "not found" means nothing;
//   ext_stop    shared stop flag: set when a flat is found, and the search
//               gives up (inconclusively) if someone else sets it;
//   stride      > 1 examines only every stride-th root candidate (a sample);
//   abort       optional second flag that makes the search give up;
//   sym         a group of linear maps permuting the direction spaces of the
//               flats in S (see group.hpp); only one root candidate per orbit
//               is examined, and the search below keeps using the stabilisers.
static bool decide(const u64* S, int n, int k, const Options& opt, Path& path, u32& leaf, Stats& st,
                   bool anchored = false, u64 work_limit = 0, bool* exhausted = nullptr,
                   std::atomic<bool>* ext_stop = nullptr, u32 stride = 1, std::atomic<bool>* abort = nullptr,
                   const SymGroup* sym = nullptr) {
    path.v.clear();
    path.p.clear();
    u64 total = popcount_arr(S, nwords(n));
    if (k == 0) {
        if (!total) return false;
        leaf = (u32)first_set(S, nwords(n));
        return true;
    }
    if (k > n || total < ((u64)1 << k)) return false;
    if (anchored && !getbit(S, 0)) return false;
    int pmax = opt.use_prefix ? n - k : n - 1;
    u32 vend = (u32)1 << (pmax + 1);  // root directions are 1 .. vend-1
    if (sym && (sym->trivial() || anchored)) sym = nullptr;  // an affine symmetry need not fix the origin
    std::vector<u32> roots;           // with symmetry: one root direction per orbit
    if (sym) {
        for (u32 v = 1; v < vend; ++v)
            if (getbit(sym->is_min.data(), v)) roots.push_back(v);
        st.sym_skipped += vend - 1 - roots.size();
    }
    const u32 nroot = sym ? (u32)roots.size() : vend - 1;
    std::atomic<u32> next(1);
    std::atomic<bool> own_stop(false);
    std::atomic<bool>& stop = ext_stop ? *ext_stop : own_stop;
    std::mutex mu;
    bool found = false;
    u64 need = (u64)1 << (k - 1);
    int nt = std::max(1, std::min<int>(opt.threads, (int)std::min<u32>(std::max<u32>(nroot, 1), 1024)));
    // With symmetry the work is very uneven: a few roots carry most of it, and
    // inside them a few second directions.  One subtree is one thread's job,
    // so the search runs in up to three passes.
    //   1: every root on a work allowance; those that exceed it are put aside;
    //   2: each heavy root again, with the full symmetry machinery and split
    //      into slices of its second direction, which all threads share; a
    //      second direction that exceeds the allowance is put aside;
    //   3: each such pair of directions, split into slices of the third.
    const bool two_pass = sym && !work_limit && opt.threads > 1 && k > 2;
    const u64 allowance = opt.sym_allowance;
    std::vector<u32> heavy;                     // roots for pass 2
    std::vector<std::pair<u32, u32>> heavy2;    // (index into heavy, second direction) for pass 3
    std::vector<SymShare> shares, shares2;      // stabiliser data of the heavy roots / pairs
    std::atomic<u32> next2(0);
    const u32 nslice = two_pass ? (u32)std::max(8, 4 * opt.threads) : 1;
    auto worker = [&](int pass) {
        Ctx ctx;
        ctx.init(n, &opt, &stop);
        ctx.anchored = anchored;
        ctx.abort = abort;
        if (sym) ctx.use_sym(sym);
        if (work_limit) ctx.work_limit = work_limit / nt + 1;
        ctx.deep_depth = !sym ? 0 : two_pass ? pass - 1 : 1;
        for (;;) {
            u32 v, v2 = 0, hidx = 0;
            if (pass == 1) {
                u32 idx = next.fetch_add(1);
                if (idx > nroot || ctx.stopped() || ctx.over_budget()) break;
                if (stride > 1 && idx % stride != stride - 1) continue;
                v = sym ? roots[idx - 1] : idx;
            } else {
                u32 task = next2.fetch_add(1);
                size_t ntask = (pass == 2 ? heavy.size() : heavy2.size()) * nslice;
                if (task >= ntask || ctx.stopped()) break;
                if (pass == 2) hidx = task / nslice;
                else {
                    hidx = heavy2[task / nslice].first;
                    v2 = heavy2[task / nslice].second;
                    ctx.share[2] = &shares2[task / nslice];
                }
                if (ctx.stopped()) break;
                v = heavy[hidx];
                ctx.share[1] = &shares[hidx];
                ctx.slice_depth = pass - 1;
                ctx.slice_n = nslice;
                ctx.slice_k = task % nslice;
                ctx.child_allow = pass == 2 && k > 4 ? allowance : 0;
            }
            if (anchored && !getbit(S, v)) continue;
            int p = msb32(v);
            if (pass == 1) ++ctx.st.cands;
            ctx.st.work += 2 * nwords(n);
            u64 cnt = quotient(S, n, v, ctx.buf[1].data());
            if (cnt < need) continue;
            if (opt.use_shadow && !top6_ok(shadow64(ctx.buf[1].data(), n - 1), n - 1, k - 1, p)) continue;
            ctx.dirv[0] = v;
            ctx.dirp[0] = p;
            if (sym && k > 1) ctx.root_low(v);
            if (two_pass && pass == 1) ctx.work_limit = ctx.st.work + allowance;
            bool hit;
            if (pass == 3) {  // replay the second direction, then search below it
                ctx.sym_enter(1, n - 1);
                ctx.gstop[1] = 0;
                u64 cnt2 = quotient(ctx.buf[1].data(), n - 1, v2, ctx.buf[2].data());
                hit = cnt2 >= need / 2 && ctx.try_child(1, v2, msb32(v2), n - 1, k - 1, cnt2);
            } else hit = ctx.descend(1, n - 1, k - 1, p, cnt);
            if (two_pass && pass == 1) {
                ctx.work_limit = ~0ULL;
                if (!hit && ctx.exhausted) {  // over the allowance: nothing is known about this root yet
                    ctx.exhausted = false;
                    std::lock_guard<std::mutex> g(mu);
                    heavy.push_back(v);
                    continue;
                }
            }
            if (pass == 2 && !ctx.heavy_kids.empty()) {
                std::lock_guard<std::mutex> g(mu);
                for (u32 w : ctx.heavy_kids) heavy2.push_back({hidx, w});
                ctx.heavy_kids.clear();
            }
            if (pass == 3) ctx.share[2]->done();
            if (hit) {
                std::lock_guard<std::mutex> g(mu);
                if (!found) {
                    found = true;
                    leaf = ctx.leaf;
                    path.v.assign(ctx.dirv, ctx.dirv + ctx.found_depth);
                    path.p.assign(ctx.dirp, ctx.dirp + ctx.found_depth);
                }
                stop.store(true);
                break;
            }
        }
        std::lock_guard<std::mutex> g(mu);
        st.add(ctx.st);
        if (ctx.exhausted && exhausted) *exhausted = true;
    };
    auto run_pass = [&](int pass, int threads) {
        if (threads == 1) worker(pass);
        else {
            std::vector<std::thread> th;
            for (int i = 0; i < threads; ++i) th.emplace_back(worker, pass);
            for (auto& t : th) t.join();
        }
    };
    auto live = [&]() { return !found && !stop.load() && !(abort && abort->load()); };
    run_pass(1, nt);
    if (two_pass && live() && !heavy.empty()) {
        std::sort(heavy.begin(), heavy.end());
        shares = std::vector<SymShare>(heavy.size());
        run_pass(2, std::max(1, opt.threads));
        if (live() && !heavy2.empty()) {
            std::sort(heavy2.begin(), heavy2.end());
            shares2 = std::vector<SymShare>(heavy2.size());
            for (SymShare& sh : shares2) sh.left.store((int)nslice);
            next2.store(0);
            run_pass(3, std::max(1, opt.threads));
        }
    }
    return found;
}

static u32 deposit(u32 z, u32 mask) {  // software pdep
    u32 r = 0;
    for (int i = 0; mask; ++i, mask &= mask - 1)
        if ((z >> i) & 1) r |= mask & -mask;
    return r;
}

// A random invertible linear change of coordinates, used for the exhaustive
// searches only.  Normality does not depend on coordinates, but the search's
// pruning does, and structure aligned with the given coordinates can make it
// markedly better or worse.
struct Recoord {
    std::vector<u32> col;        // images of the unit vectors
    std::vector<u64> S[2];       // S[c](x) = original S[c](M x)
    bool ready = false;
    u32 apply(u32 x) const {
        u32 r = 0;
        for (; x; x &= x - 1) r ^= col[__builtin_ctz(x)];
        return r;
    }
    void build(const std::vector<u64>* orig, int m, u64 seed) {
        std::mt19937_64 rng(seed);
        u32 lim = m >= 32 ? ~0u : ((1u << m) - 1);
        std::vector<u32> cols;
        for (;;) {
            cols.assign(m, 0);
            std::vector<u32> bas(m, 0);
            int rank = 0;
            for (int i = 0; i < m; ++i) {
                u32 v = cols[i] = (u32)rng() & lim;
                while (v) {
                    int h = msb32(v);
                    if (!bas[h]) { bas[h] = v; ++rank; break; }
                    v ^= bas[h];
                }
            }
            if (rank == m) break;
        }
        build(orig, m, cols);
    }
    void build(const std::vector<u64>* orig, int m, const std::vector<u32>& cols) {  // cols: an invertible matrix
        col = cols;
        for (int c = 0; c < 2; ++c) S[c].assign(nwords(m), 0);
        u32 img = 0;
        for (u32 g = 0;; ++g) {  // walk x in Gray order, img = M x
            u32 x = g ^ (g >> 1);
            S[getbit(orig[1].data(), img)][x >> 6] |= 1ULL << (x & 63);
            if (g + 1 == ((u32)1 << m)) break;
            img ^= col[__builtin_ctz(g + 1)];
        }
        ready = true;
    }
};

// F: truth table bitset of f on n variables (bit x = f(x)).
// With target >= 0 only the question "is f target-normal?" is decided: R.k is
// then >= target if yes, and the best flat found (a lower bound) if no.
static Result solve(const std::vector<u64>& F, int n, const Options& opt, int greedy_restarts = 3, int target = -1) {
    double t0 = now_sec();
    Result R;
    R.n = n;
    size_t N = (size_t)1 << n;
    u64 wt = popcount_arr(F.data(), nwords(n));
    if (wt == 0 || wt == N) {  // constant
        R.k = n;
        R.value = wt ? 1 : 0;
        for (int i = 0; i < n; ++i) R.basis.push_back(1u << i);
        R.ub[R.value] = n;
        R.seconds = now_sec() - t0;
        return R;
    }
    // --- invariant linear structures ---------------------------------------
    std::vector<int64_t> sq(N);
    {
        std::vector<int32_t> W(N);
        for (size_t x = 0; x < N; ++x) W[x] = getbit(F.data(), (u32)x) ? -1 : 1;
        wht(W.data(), n);
        for (size_t x = 0; x < N; ++x) sq[x] = (int64_t)W[x] * W[x];
        wht(sq.data(), n);  // sq[v] = 2^n * sum_x (-1)^(f(x)+f(x+v))
    }
    std::vector<u32> lb(n, 0);
    int64_t full = (int64_t)1 << (2 * n);
    for (size_t x = 1; x < N; ++x) {
        if (sq[x] != full) continue;
        u32 v = (u32)x;
        while (v) {
            int b = msb32(v);
            if (!lb[b]) { lb[b] = v; break; }
            v ^= lb[b];
        }
    }
    std::vector<int64_t>().swap(sq);
    u32 keep = 0;
    for (int i = 0; i < n; ++i)
        if (lb[i]) ++R.lin_dim; else keep |= 1u << i;
    int m = n - R.lin_dim;
    size_t M = (size_t)1 << m;
    std::vector<u64> S[2];
    S[0].assign(nwords(m), 0);
    S[1].assign(nwords(m), 0);
    for (size_t z = 0; z < M; ++z)
        S[getbit(F.data(), deposit((u32)z, keep))][z >> 6] |= 1ULL << (z & 63);
    u64 size[2] = {popcount_arr(S[0].data(), nwords(m)), popcount_arr(S[1].data(), nwords(m))};
    // --- spectral upper bounds ---------------------------------------------
    // f == c on a k-flat a+E  =>  sum over u in E-perp, u != 0, of |W(u)| >= 2|S_(1-c)|,
    // a sum of 2^(m-k) - 1 Walsh coefficients.
    {
        std::vector<int32_t> W(M);
        for (size_t z = 0; z < M; ++z) W[z] = getbit(S[1].data(), (u32)z) ? -1 : 1;
        wht(W.data(), m);
        std::vector<int32_t> ab(M - 1);
        for (size_t z = 1; z < M; ++z) ab[z - 1] = W[z] < 0 ? -W[z] : W[z];
        std::sort(ab.begin(), ab.end(), [](int32_t x, int32_t y) { return x > y; });
        std::vector<int64_t> pre(M, 0);
        for (size_t i = 0; i + 1 < M; ++i) pre[i + 1] = pre[i] + ab[i];
        for (int c = 0; c < 2; ++c) {
            int k = 0;
            while (k < m && ((u64)2 << k) <= size[c]) ++k;
            for (; k > 0; --k)
                if (pre[((size_t)1 << (m - k)) - 1] >= (int64_t)(2 * size[1 - c])) break;
            R.ub[c] = k;
        }
    }
    // --- greedy lower bound --------------------------------------------------
    Flat best;  // in the reduced coordinates
    int B = -1;
    std::mt19937_64 rng(12345);
    for (int c = 0; c < 2; ++c)
        for (int it = 0; it <= greedy_restarts; ++it) {
            Path p;
            u32 leaf = greedy_extend(S[c], m, p, it ? &rng : nullptr);
            if ((int)p.v.size() > B) { B = (int)p.v.size(); lift(p, leaf, best); best.c = c; }
            if (B >= std::max(R.ub[0], R.ub[1])) break;
        }
    R.greedy_k = B + R.lin_dim;
    if (opt.verbose)
        fprintf(stderr, "[n=%d] linear structures: %d, reduced n=%d, upper bounds c0:%d c1:%d, greedy: %d  (%.3fs)\n",
                n, R.lin_dim, m, R.ub[0], R.ub[1], B, now_sec() - t0);
    const int tt = target >= 0 ? target - R.lin_dim : -1;  // target in the reduced problem
    // --- flats through the origin --------------------------------------------
    // Structured functions are often constant on a large *subspace*.  Searching
    // only those is far cheaper (few admissible directions), so it is tried
    // first.  It only supplies lower bounds, so it runs on a fixed work budget
    // (single-threaded, which also keeps it deterministic).
    {
        int c = getbit(S[1].data(), 0);
        u64 budget = std::max((u64)512 << m, (u64)1 << 26);
        Options aopt = opt;
        aopt.threads = 1;
        Stats ast;
        int t_hi = target >= 0 ? std::min(tt, R.ub[c]) : R.ub[c];  // fixed target: only that dimension
        int t_lo = target >= 0 ? std::max(B, tt - 1) : B;
        for (int t = t_hi; t > t_lo && ast.work < budget; --t) {
            Path p;
            u32 leaf = 0;
            double ts = now_sec();
            bool cut = false;
            bool ok = decide(S[c].data(), m, t, aopt, p, leaf, ast, true, budget - ast.work, &cut);
            if (opt.verbose)
                fprintf(stderr, "  subspace search: %d-dim subspace with f=%d: %s  (%.3fs, %.1f%% of budget used)\n",
                        t, c, ok ? "FOUND" : cut ? "budget exhausted" : "none", now_sec() - ts,
                        100.0 * (double)ast.work / (double)budget);
            if (ok) {
                leaf = replay_extend(S[c].data(), m, p);
                B = (int)p.v.size();
                lift(p, leaf, best);
                best.c = c;
                R.anchored_k = B + R.lin_dim;
                break;
            }
        }
    }
    // --- symmetry ------------------------------------------------------------
    // Looked at only when an exhaustive search is needed.  Supplied candidates
    // come in the original coordinates and are carried to the reduced
    // function; they are checked when the first search is due.  If they give
    // nothing, the library (if any) is started on the reduced function and
    // runs alongside the searches: its answer is picked up when it arrives,
    // and a search that is running at that moment starts over with the group.
    // Every map is checked on all 2^m points before it is used.
    SymGroup symA, symB;                 // the group in the given and in the adapted coordinates
    Recoord rc, rcg;                     // random coordinates; coordinates adapted to the group
    std::atomic<bool> sym_ready(false);  // symA is final and may be read
    std::atomic<bool> sym_flip(false);   // some map takes f to f + 1
    std::atomic<bool> lib_open(false);   // the library is running and its answer is still to come
    std::unique_ptr<AutJob> job;
    std::mutex mu, mu_sym;
    std::vector<LMap> keepers, flips;    // linear parts of the verified maps
    auto check_maps = [&](const std::vector<AffMap>& cand) {
        for (const AffMap& a : cand) {
            if ((int)a.col.size() != m) continue;
            LMap M;
            for (int i = 0; i < 32; ++i) M.c[i] = i < m ? a.col[i] : 0;
            int kind = aut_kind(S[1].data(), m, M, a.t);
            if (kind == 0) keepers.push_back(M);
            else if (kind == 1) flips.push_back(M);
        }
        R.sym_offered += (int)cand.size();
    };
    auto finish_sym = [&](double ts) {
        symA.m = m;
        symA.gens = keepers;
        if (!flips.empty()) {  // the maps with f(Ax) = f(x) have index 2: Schreier generators
            LMap ci;
            lm_inverse(flips[0], m, ci);
            for (const LMap& g : keepers) symA.gens.push_back(lm_mul(flips[0], lm_mul(g, ci, m), m));
            for (const LMap& f : flips) {
                symA.gens.push_back(lm_mul(flips[0], f, m));
                symA.gens.push_back(lm_mul(f, ci, m));
            }
        }
        sym_finish(symA);
        R.sym_loaded = true;
        R.sym_flip = !flips.empty();
        R.sym_gens = (int)symA.gens.size();
        R.sym_order = symA.trivial() ? 1 : symA.full ? (double)symA.all.size() + 1 : 0;
        R.sym_seconds += now_sec() - ts;
        if (opt.verbose) {
            fprintf(stderr, "  symmetry: %d maps offered, %d generators kept", R.sym_offered, R.sym_gens);
            if (R.sym_order >= 1) fprintf(stderr, ", group of order %.0f", R.sym_order);
            else fprintf(stderr, ", group of order above %zu", SYM_ROOT_CAP);
            fprintf(stderr, "%s  (%.3fs)%s%s\n", R.sym_flip ? ", f ~ f+1" : "", now_sec() - ts,
                    R.sym_note.empty() ? "" : "  ", R.sym_note.c_str());
        }
        sym_flip.store(R.sym_flip);
        sym_ready.store(true);
    };
    auto load_cands = [&]() {
        double ts = now_sec();
        auto red = [&](u32 x) {  // class of x modulo the linear structures, in reduced coordinates
            for (int i = n - 1; i >= 0; --i)
                if (lb[i] && ((x >> i) & 1)) x ^= lb[i];
            u32 z = 0;
            int o = 0;
            for (u32 k = keep; k; k &= k - 1, ++o)
                if (x & k & -k) z |= 1u << o;
            return z;
        };
        std::vector<AffMap> cand;
        for (const AffMap& a : *opt.sym_cands) {
            if ((int)a.col.size() != n) continue;
            auto img = [&](u32 z) {
                u32 x = deposit(z, keep), y = a.t;
                for (; x; x &= x - 1) y ^= a.col[__builtin_ctz(x)];
                return red(y);
            };
            AffMap b;
            b.t = img(0);
            for (int i = 0; i < m; ++i) b.col.push_back(img(1u << i) ^ b.t);
            cand.push_back(b);
        }
        check_maps(cand);
        R.sym_loaded = true;
        if (!keepers.empty() || !flips.empty()) {  // enough to go on: the library is not asked
            finish_sym(ts);
            lib_open.store(false);
        }
    };
    // Has the library answered?  If so its maps are checked and the group is
    // built.  Returns whether the question is settled now (by this call).
    auto poll_library = [&]() -> bool {
        if (!lib_open.load()) return false;
        std::lock_guard<std::mutex> g(mu_sym);
        if (!lib_open.load()) return false;
        std::vector<AffMap> lib;
        std::string note;
        int st = job->poll(lib, note);
        if (st == 0) return false;
        double ts = now_sec();
        R.sym_note = note;
        if (st == 1) check_maps(lib);
        finish_sym(ts);
        lib_open.store(false);
        return true;
    };
    // --- exhaustive searches -------------------------------------------------
    // Search for a t-flat on which f == c.  A sample of the root candidates is
    // first searched in the given coordinates and, if that is not quick, raced
    // against a random re-coordinatisation; the full search runs in the winner.
    // With a symmetry group the alternative coordinates are not random but
    // adapted to it (few orbits in the low coordinates), and both use the group.
    // `second`: the other constant is being searched at the same time; if a map
    // taking f to f + 1 turns up, this search is redundant and is given up.
    std::function<bool(int, int, int, std::atomic<bool>*, Flat&, bool)> search_one =
        [&](int c, int t, int nt, std::atomic<bool>* stop, Flat& out, bool second) -> bool {
        Options o = opt;
        o.threads = std::max(1, nt);
        poll_library();
        const bool useg = sym_ready.load() && !symA.trivial();
        Recoord& arc = useg ? rcg : rc;  // the alternative coordinates
        auto prepare_alt = [&]() {
            std::lock_guard<std::mutex> g(mu);
            if (arc.ready) return;
            if (useg) {
                double ts = now_sec();
                rcg.build(S, m, sym_adapted_basis(symA.gens, m));
                symB = sym_conjugate(symA, rcg.col);
                R.sym_seconds += now_sec() - ts;
            } else rc.build(S, m, 20240229);
        };
        auto run = [&](bool scr, u32 stride, std::atomic<bool>* abort) -> bool {
            const u64* Sx = scr ? arc.S[c].data() : S[c].data();
            const SymGroup* grp = useg ? (scr ? &symB : &symA) : nullptr;
            Path p;
            u32 leaf = 0;
            Stats s;
            bool ok = decide(Sx, m, t, o, p, leaf, s, false, 0, nullptr, stop, stride, abort, grp);
            Flat fl;
            if (ok) {
                leaf = replay_extend(Sx, m, p);
                lift(p, leaf, fl);
                fl.c = c;
                if (scr) {
                    fl.a = arc.apply(fl.a);
                    for (u32& b : fl.basis) b = arc.apply(b);
                }
            }
            std::lock_guard<std::mutex> g(mu);
            R.st.add(s);
            if (ok) out = fl;
            return ok;
        };
        double ts = now_sec();
        bool scr = false;
        int pmax = opt.use_prefix ? m - t : m - 1;
        const u32 vend = pmax >= 0 && pmax < 31 ? (u32)2 << pmax : 1;
        // root candidates and sampling stride in the given (a) and the alternative (b) coordinates
        u32 na = useg ? symA.count_roots(vend) : vend - 1, sa = na / 32;
        u32 nb = na, sb = sa;
        const bool alt_ok = !useg || opt.sym_coords;
        auto alt_counts = [&]() {
            prepare_alt();
            if (useg) { nb = symB.count_roots(vend); sb = std::max<u32>(nb / 32, 1); }
        };
        const char* alt = useg ? "adapted coordinates" : "random coordinates";
        std::string how = "given coordinates";
        if (useg && alt_ok && sa >= 4) {
            // With symmetry the cost sits in the first few roots, which a sample
            // by stride misses, so the two systems cannot be compared by
            // sampling.  A sample that finishes at once means the search is
            // short; otherwise the coordinates made for the group are used.
            using namespace std::chrono;
            std::atomic<bool> abort_a(false);
            double a0 = now_sec();
            auto fa = std::async(std::launch::async, [&] { return run(false, sa, &abort_a); });
            bool quick = fa.wait_for(milliseconds(100)) == std::future_status::ready;
            if (!quick) abort_a.store(true);
            if (fa.get()) return true;
            if (stop->load()) return false;
            if (!quick || (now_sec() - a0) * sa > 0.5) {
                alt_counts();
                scr = true;
                how = alt;
            }
        } else if (sa >= 4 && alt_ok) {
            using namespace std::chrono;
            std::atomic<bool> abort_a(false), abort_b(false);
            double a0 = now_sec();
            auto fa = std::async(std::launch::async, [&] { return run(false, sa, &abort_a); });
            bool a_done = fa.wait_for(milliseconds(200)) == std::future_status::ready;
            if (a_done) {
                if (fa.get()) return true;
                double ta = (now_sec() - a0) * sa;  // predicted time of the full search
                if (stop->load()) return false;
                if (ta > 1.0) {  // long enough to be worth a second opinion
                    alt_counts();
                    double b0 = now_sec();
                    if (run(true, sb, nullptr)) return true;
                    double tb = (now_sec() - b0) * sb;
                    if (stop->load()) return false;
                    scr = tb < 0.7 * ta;
                    how = scr ? alt : "given coordinates (sampled both)";
                }
            } else {  // slow: race a sample in the alternative coordinates, keep the better prediction
                alt_counts();
                double b0 = now_sec();
                auto fb = std::async(std::launch::async, [&] { return run(true, sb, &abort_b); });
                double pa = -1, pb = -1;
                for (;;) {
                    if (pa < 0 && fa.wait_for(milliseconds(3)) == std::future_status::ready) pa = (now_sec() - a0) * sa;
                    if (pb < 0 && fb.wait_for(milliseconds(3)) == std::future_status::ready) pb = (now_sec() - b0) * sb;
                    if (pa >= 0 && pb >= 0) break;
                    double now = now_sec();
                    if (pa >= 0 && (now - b0) * sb >= pa) { abort_b.store(true); break; }
                    if (pb >= 0 && (now - a0) * sa >= pb) { abort_a.store(true); break; }
                }
                bool ra = fa.get(), rb = fb.get();
                if (ra || rb) return true;
                if (stop->load()) return false;
                scr = pb >= 0 && (pa < 0 || pb < pa);
                how = std::string(scr ? alt : "given coordinates") + " (won the race)";
            }
        } else if (useg && alt_ok) {  // too few root candidates to sample: take the system with fewer of them
            alt_counts();
            scr = nb < na;
            how = scr ? alt : "given coordinates";
        }
        bool ok;
        if (!useg) {  // did the library answer while the samples were running?
            poll_library();
            if (second && sym_flip.load()) return false;
            if (sym_ready.load() && !symA.trivial()) return search_one(c, t, nt, stop, out, second);
        }
        if (!useg && lib_open.load()) {
            // The library is still at work.  Search meanwhile, but if a group
            // arrives, start this search over with it.
            using namespace std::chrono;
            std::atomic<bool> abort_f(false);
            auto ff = std::async(std::launch::async, [&] { return run(scr, 1, &abort_f); });
            bool restart = false, redundant = false;
            while (ff.wait_for(milliseconds(20)) != std::future_status::ready) {
                poll_library();
                if (second && sym_flip.load()) {
                    abort_f.store(true);
                    redundant = true;
                    break;
                }
                if (sym_ready.load() && !symA.trivial()) {
                    abort_f.store(true);
                    restart = true;
                    break;
                }
                if (!lib_open.load()) break;  // settled, and no group: just finish
            }
            ok = ff.get();
            if (!ok && redundant) return false;
            if (!ok && restart) {
                if (stop->load()) return false;
                return search_one(c, t, nt, stop, out, second);
            }
        } else ok = run(scr, 1, nullptr);
        if (opt.verbose) {
            std::lock_guard<std::mutex> g(mu);
            fprintf(stderr, "  search: %d-flat with f=%d: %s  (%.3fs, %s", t, c,
                    ok ? "FOUND" : stop->load() ? "stopped" : "none", now_sec() - ts, how.c_str());
            if (useg) fprintf(stderr, ", %u of %u root directions", scr ? nb : na, vend - 1);
            fprintf(stderr, ")\n");
        }
        return ok;
    };
    // Both constants at once when both are possible: a quick success for one
    // should not have to wait for a slow refutation of the other.
    bool cands_seen = false;
    auto try_dim = [&](int t) -> bool {
        int cs[2], nc = 0;
        for (int cc = 0; cc < 2; ++cc) {
            int c = R.ub[0] >= R.ub[1] ? cc : 1 - cc;
            if (R.ub[c] >= t) cs[nc++] = c;
        }
        if (!cands_seen) {
            cands_seen = true;
            if (opt.sym_cands && !opt.sym_cands->empty()) load_cands();
            if (!sym_ready.load() && opt.aut_source && opt.aut_budget > 0) {  // nothing usable yet: start the library
                job.reset(opt.aut_source(S[1].data(), m, opt.aut_budget));
                if (job) lib_open.store(true);
                else R.sym_note = "library: could not be started";
                R.sym_loaded = true;
            }
        }
        poll_library();
        if (sym_flip.load() && nc == 2) nc = 1;  // an automorphism swaps f = 0 and f = 1: one constant decides both
        std::atomic<bool> stop(false);
        Flat fl[2];
        bool ok[2] = {false, false};
        R.searches += nc;
        if (nc == 2 && opt.threads >= 2) {
            std::thread other([&] { ok[1] = search_one(cs[1], t, opt.threads - opt.threads / 2, &stop, fl[1], true); });
            ok[0] = search_one(cs[0], t, opt.threads / 2, &stop, fl[0], false);
            other.join();
        } else {
            for (int i = 0; i < nc && !ok[0] && !ok[1]; ++i) {
                if (i == 1 && sym_flip.load()) break;  // decided by the first constant
                ok[i] = search_one(cs[i], t, opt.threads, &stop, fl[i], false);
            }
        }
        for (int i = 0; i < 2; ++i)
            if (ok[i] && fl[i].dim() > B) { B = fl[i].dim(); best = fl[i]; }
        return ok[0] || ok[1];
    };
    if (target >= 0) {
        if (B < tt) try_dim(tt);
    } else {
        // Top-down: refuting a large flat is cheap and gets dearer as t falls, so
        // start at the spectral bound and stop at the first t that succeeds (or
        // at the lower bound).  No search is ever run below the true maximum.
        for (int t = std::max(R.ub[0], R.ub[1]); t > B; --t)
            if (try_dim(t)) break;
    }
    if (lib_open.load()) {  // the searches are over before the library was
        std::lock_guard<std::mutex> g(mu_sym);
        job->stop(R.sym_note);
        lib_open.store(false);
    }
    job.reset();
    // --- witness in the original coordinates ---------------------------------
    R.a = deposit(best.a, keep);
    for (u32 b : best.basis) R.basis.push_back(deposit(b, keep));
    for (int i = 0; i < n; ++i)
        if (lb[i]) R.basis.push_back(lb[i]);
    R.k = B + R.lin_dim;
    R.value = best.c;
    R.ub[0] += R.lin_dim;
    R.ub[1] += R.lin_dim;
    bool good = constant_on_flat(F.data(), R.a, R.basis, R.value);
    if (good) {  // the basis must be independent, or the flat is smaller than claimed
        std::vector<u32> bas(n, 0);
        for (u32 v : R.basis) {
            while (v) {
                int h = msb32(v);
                if (!bas[h]) { bas[h] = v; break; }
                v ^= bas[h];
            }
            good &= v != 0;
        }
    }
    if (!good) {
        fprintf(stderr, "INTERNAL ERROR: witness flat failed verification\n");
        R.k = -2;
    }
    R.seconds = now_sec() - t0;
    return R;
}
