// normality: largest k such that a Boolean function is k-normal
// (constant on some k-dimensional affine subspace of F_2^n), and the same
// for functions F_p^n -> F_p, p an odd prime (option --p).
//
// Build:  g++ -O3 -march=native -std=c++17 -pthread -o normality main.cpp
// (see README.md for the build with the external equivalence library)
#include "tests.hpp"
#include "autlib.hpp"
#include <fstream>
#include <functional>
#include <sstream>

static void usage() {
    puts("usage: normality [options] <truth table>\n"
         "  <truth table>    2^n characters 0/1; character i is f(i), x_j = bit j of i\n"
         "  --hex            the table is hexadecimal: one integer whose bit i is f(i)\n"
         "  -f FILE          read the table from FILE\n"
         "  -t N             worker threads (default: all cores)\n"
         "  -v               progress on stderr\n"
         "  --json           print the result as one JSON object (used by normality.py)\n"
         "built-in families (instead of a table):\n"
         "  --kasami N [K]       Kasami bent function Tr(a x^(4^K-2^K+1)), a of order 3 (K = 3: x^57)\n"
         "  --dillon N           Dillon bent function Tr(a x^(2^(N/2)-1)), a a Kloosterman zero\n"
         "  --d0 N [SEED]        Maiorana-McFarland class D_0 bent function, scrambled\n"
         "  --power N D [E]      Tr(g^E x^D) over GF(2^N), g primitive (E defaults to 0)\n"
         "  --inverse N          Tr(x^(2^N-2))\n"
         "  --aes MASK           component <MASK, S(x)> of the AES S-box\n"
         "  --mm N [SEED]        Maiorana-McFarland bent function, scrambled\n"
         "  --ps N [SEED]        partial-spread (PS_ap) bent function, scrambled\n"
         "  --cf N               Carlet-Feng function\n"
         "  --maj N              majority;   --threshold N T   (wt(x) >= T)\n"
         "  --hwb N              hidden weighted bit function\n"
         "  --random N [SEED]    uniformly random function\n"
         "modifiers:\n"
         "  --dual           replace a bent function by its dual\n"
         "  --plus U         then add the linear function <U, x>\n"
         "  --hyp R          then add R hyperbolic pairs y1 y2 + ... (2R more variables)\n"
         "  --scramble SEED  then apply a random affine change of variables\n"
         "  --all-linear     solve f + <u, x> for every u and print the distribution of k\n"
         "  --target K       only decide whether f is K-normal (much cheaper than the maximum)\n"
         "symmetry (automorphisms A of f, f(Ax) = f(x) or f(x) + 1; every map is verified before use):\n"
         "  --aut [SEC]      compute the automorphism group with the external equivalence library, in the\n"
         "                   background, for at most SEC seconds (default 60); needs a build with the library\n"
         "  --aut-known      use the symmetries known from the construction of a built-in family\n"
         "  --aut-file FILE  read candidate maps: one per line, the images of e_0 .. e_(n-1), then A(0)\n"
         "  --sym-given      keep the given coordinates (do not try coordinates adapted to the group)\n"
         "  --no-mates       use the whole group for the first direction only\n"
         "odd characteristic (f : F_p^n -> F_p, P a prime up to 61):\n"
         "  --p P            the table has P^n values 0 .. P-1; value number x_0 + x_1 P + ... is f(x).\n"
         "                   P <= 7: one digit per value; larger P: numbers separated by spaces or commas\n"
         "                   with --p, instead of a table:\n"
         "    --power N D [E]    Tr(g^E x^D) over GF(P^N), g primitive; may be repeated, the terms are added\n"
         "    --mm N [SEED]      Maiorana-McFarland bent function\n"
         "    --quad N [T]       quadratic form x0 x1 + x2 x3 + ... (T = 1: the elliptic one, x0^2 - c x1^2 + ...)\n"
         "    --random N [SEED]  uniformly random function\n"
         "                   and of the options above: -f, -t, -v, --json, --scramble, --target, and\n"
         "    --aut-known        use the maps x -> a x^(P^i) of the field that are symmetries of a --power function\n"
         "    --aut-file FILE    candidate maps A with f(A x) = f(x), or = s(f(x)) for a permutation s of the\n"
         "                       values: one per line, the images of e_0 .. e_(n-1), then A(0), as point numbers\n"
         "other:\n"
         "  --suite [--big]  run the built-in collection of structured functions (--big adds the slow ones);\n"
         "                   --suite-bent / --suite-other run one half of it\n"
         "  --selftest       run the correctness tests\n"
         "  --no-sparse --no-shadow --no-prefix --no-quick   switch individual prunings off");
}

static void print_search_stats(const Result& r) {
    printf("%d exhaustive searches (%llu dense nodes, %llu sparse nodes, %llu candidates)\n", r.searches,
           (unsigned long long)r.st.dense_nodes, (unsigned long long)r.st.sparse_nodes,
           (unsigned long long)r.st.cands);
}

static void print_symmetry(const Result& r) {
    if (!r.sym_loaded) return;
    if (r.sym_gens == 0 && !r.sym_flip) {
        printf("symmetry: none usable (%d maps offered)%s%s\n", r.sym_offered, r.sym_note.empty() ? "" : "; ",
               r.sym_note.c_str());
        return;
    }
    printf("symmetry: ");
    if (r.sym_gens == 0) printf("no automorphism with f(Ax) = f(x)");
    else if (r.sym_order >= 1) printf("group of order %.0f", r.sym_order);
    else printf("group of order above %zu", SYM_ROOT_CAP);
    if (r.sym_gens) printf(" (%d generators)", r.sym_gens);
    if (r.sym_flip) printf(", and f is equivalent to f + 1, so one constant was searched");
    printf("; %llu candidate directions skipped; %.3f s to prepare%s%s\n", (unsigned long long)r.st.sym_skipped,
           r.sym_seconds, r.sym_note.empty() ? "" : "; ", r.sym_note.c_str());
}

static void report(const Result& r) {
    printf("n = %d\n", r.n);
    printf("largest k such that f is k-normal: %d\n", r.k);
    printf("witness: f = %d on a + span(basis), a = 0x%x, basis = {", r.value, r.a);
    for (size_t i = 0; i < r.basis.size(); ++i) printf("%s0x%x", i ? ", " : "", r.basis[i]);
    printf("}\n");
    printf("spectral upper bounds: f=0: %d, f=1: %d;  greedy lower bound: %d\n", r.ub[0], r.ub[1], r.greedy_k);
    if (r.lin_dim) printf("invariant linear structures factored out: %d\n", r.lin_dim);
    if (r.anchored_k >= 0) printf("lower bound improved to %d by the search over subspaces through the origin\n", r.anchored_k);
    if (r.searches == 0) printf("optimality: the flat found meets the spectral bound, no exhaustive search needed\n");
    else {
        printf("optimality: ");
        print_search_stats(r);
    }
    print_symmetry(r);
    printf("time: %.3f s\n", r.seconds);
}

// The result as one JSON object on stdout, for programs (normality.py reads it).
static void print_json(const Result& r, int n, int target, const TT& F) {
    long long mw = max_walsh(F, n);
    bool bent = n % 2 == 0 && mw * mw == (1ll << n);
    auto esc = [](const std::string& t) {
        std::string o;
        for (char ch : t) {
            if (ch == '"' || ch == '\\') o.push_back('\\');
            if ((unsigned char)ch >= 32) o.push_back(ch);
        }
        return o;
    };
    printf("{\"p\": 2, \"n\": %d, \"k\": %d, \"exact\": %s, \"value\": %d, \"offset\": %u, \"basis\": [", n, r.k,
           target < 0 ? "true" : "false", r.value, r.a);
    for (size_t i = 0; i < r.basis.size(); ++i) printf("%s%u", i ? ", " : "", r.basis[i]);
    printf("], \"upper_bounds\": [%d, %d], \"greedy\": %d, \"linear_structures\": %d", r.ub[0], r.ub[1], r.greedy_k,
           r.lin_dim);
    if (target >= 0) printf(", \"target\": %d, \"target_normal\": %s", target, r.k >= target ? "true" : "false");
    printf(", \"max_walsh\": %lld, \"bent\": %s", mw, bent ? "true" : "false");
    printf(", \"searches\": %d, \"dense_nodes\": %llu, \"sparse_nodes\": %llu, \"candidates\": %llu", r.searches,
           (unsigned long long)r.st.dense_nodes, (unsigned long long)r.st.sparse_nodes, (unsigned long long)r.st.cands);
    printf(", \"symmetry\": {\"looked_at\": %s, \"generators\": %d, \"order\": %.0f, \"flip\": %s, \"skipped\": %llu, "
           "\"note\": \"%s\"}",
           r.sym_loaded ? "true" : "false", r.sym_gens, r.sym_order, r.sym_flip ? "true" : "false",
           (unsigned long long)r.st.sym_skipped, esc(r.sym_note).c_str());
    printf(", \"seconds\": %.6f}\n", r.seconds);
}

// ------------------------------------------------------------ odd characteristic
static bool read_maps(const char* path, int n, std::vector<AffMap>& out);

static std::string pa_vec(u32 x, int p, int n) {  // the coordinates x_0 .. x_(n-1)
    std::string o;
    for (int i = 0; i < n; ++i, x /= p) {
        if (p > 7 && i) o.push_back(' ');
        o += std::to_string(x % p);
    }
    return o;
}

static void pa_report(const PResult& r) {
    printf("p = %d, n = %d\n", r.p, r.n);
    printf("largest k such that f is k-normal: %d\n", r.k);
    printf("witness: f = %d on a + span(basis), points given as x_0 + x_1 p + ... and by their coordinates x_0 .. x_%d\n",
           r.value, r.n - 1);
    printf("  a     = %-10u (%s)\n", r.a, pa_vec(r.a, r.p, r.n).c_str());
    for (size_t i = 0; i < r.basis.size(); ++i)
        printf("  %s %-10u (%s)\n", i ? "       " : "basis =", r.basis[i], pa_vec(r.basis[i], r.p, r.n).c_str());
    printf("spectral upper bounds:");
    for (int c = 0; c < r.p; ++c) printf(" f=%d: %d%s", c, r.ub[c], c + 1 < r.p ? "," : "");
    printf(";  greedy lower bound: %d\n", r.greedy_k);
    if (r.lin_dim) printf("invariant linear structures factored out: %d\n", r.lin_dim);
    if (r.searches == 0) printf("optimality: the flat found meets the spectral bound, no exhaustive search needed\n");
    else
        printf("optimality: %d exhaustive searches (%llu dense nodes, %llu sparse nodes, %llu candidates)\n", r.searches,
               (unsigned long long)r.st.dense_nodes, (unsigned long long)r.st.sparse_nodes, (unsigned long long)r.st.cands);
    if (r.sym_looked) {
        printf("symmetry: %d of %d maps are symmetries", r.sym_used, r.sym_offered);
        if (r.sym_used)
            printf("; %d classes of constants, one constant of each searched; %llu first directions skipped",
                   r.sym_classes, (unsigned long long)r.st.sym_skipped);
        printf("%s%s\n", r.sym_note.empty() ? "" : "; ", r.sym_note.c_str());
    }
    printf("time: %.3f s\n", r.seconds);
}

static void pa_print_json(const PResult& r, int target) {
    printf("{\"p\": %d, \"n\": %d, \"k\": %d, \"exact\": %s, \"value\": %d, \"offset\": %u, \"basis\": [", r.p, r.n, r.k,
           target < 0 ? "true" : "false", r.value, r.a);
    for (size_t i = 0; i < r.basis.size(); ++i) printf("%s%u", i ? ", " : "", r.basis[i]);
    printf("], \"upper_bounds\": [");
    for (int c = 0; c < r.p; ++c) printf("%s%d", c ? ", " : "", r.ub[c]);
    printf("], \"greedy\": %d, \"linear_structures\": %d", r.greedy_k, r.lin_dim);
    if (target >= 0) printf(", \"target\": %d, \"target_normal\": %s", target, r.k >= target ? "true" : "false");
    printf(", \"max_walsh\": %.9g, \"bent\": %s", r.max_walsh, r.bent ? "true" : "false");
    printf(", \"searches\": %d, \"dense_nodes\": %llu, \"sparse_nodes\": %llu, \"candidates\": %llu", r.searches,
           (unsigned long long)r.st.dense_nodes, (unsigned long long)r.st.sparse_nodes, (unsigned long long)r.st.cands);
    int values = 0;
    for (int c = 0; c < r.p; ++c) values += r.ub[c] >= 0;
    printf(", \"symmetry\": {\"looked_at\": %s, \"generators\": %d, \"order\": %.0f, \"flip\": %s, \"skipped\": %llu, "
           "\"classes\": %d, \"note\": \"%s\"}",
           r.sym_looked ? "true" : "false", r.sym_used, r.sym_order, r.sym_used && r.sym_classes < values ? "true" : "false",
           (unsigned long long)r.st.sym_skipped, r.sym_classes, r.sym_note.c_str());
    printf(", \"seconds\": %.6f}\n", r.seconds);
}

static int pary_main(int argc, char** argv, int p) {
    PField F;
    if (!F.init(p)) { fprintf(stderr, "--p needs a prime between 2 and 61\n"); return 2; }
    Options opt;
    opt.threads = std::max(1u, std::thread::hardware_concurrency());
    bool json = false, known = false, field = false;
    int target = -1, mode = 0, n = -1;
    u64 scr = 0;
    std::string table, aut_file;
    PT f;
    auto fits = [&](int nn) { return nn >= 1 && nn <= F.maxdim && F.pw[nn] <= 60000000; };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto num = [&](int j) { return j < argc ? strtoull(argv[j], nullptr, 0) : 0ULL; };
        auto has = [&](int j) { return j < argc && isdigit((unsigned char)argv[j][0]); };
        if (a == "--p" && has(i + 1)) ++i;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--json") json = true;
        else if (a == "-v") opt.verbose = true;
        else if (a == "--target" && has(i + 1)) target = (int)num(++i);
        else if (a == "--scramble" && has(i + 1)) scr = num(++i);
        else if (a == "-t" && has(i + 1)) opt.threads = std::max(1, (int)num(++i));
        else if (a == "--sparse-weight" && i + 1 < argc) opt.sparse_weight = atof(argv[++i]);
        else if (a == "--p-mode" && has(i + 1)) mode = (int)num(++i);  // 1: lists only, 2: bitsets wherever possible
        else if (a == "--aut-known") known = true;
        else if (a == "--aut-file" && i + 1 < argc) aut_file = argv[++i];
        else if (a == "-f" && i + 1 < argc) {
            std::ifstream in(argv[++i]);
            if (!in) { fprintf(stderr, "cannot open %s\n", argv[i]); return 2; }
            std::stringstream ss;
            ss << in.rdbuf();
            table = ss.str();
        } else if ((a == "--random" || a == "--mm" || a == "--quad") && has(i + 1)) {
            n = (int)num(++i);
            const u64 arg = has(i + 1) ? num(++i) : a == "--quad" ? 0 : 1;
            if (!fits(n) || (a == "--mm" && n % 2) || (a == "--quad" && arg == 1 && (n % 2 || p == 2))) {
                fprintf(stderr, "%s: no such function for p = %d, n = %d\n", a.c_str(), p, n);
                return 2;
            }
            f = a == "--random" ? pa_random(p, n, arg) : a == "--mm" ? pa_mm(p, n, arg) : pa_quadratic(p, n, (int)arg);
        } else if (a == "--power" && has(i + 2)) {
            const int nn = (int)num(++i);
            const u64 d = num(++i), e = has(i + 1) ? num(++i) : 0;
            if (!fits(nn) || (n >= 0 && nn != n) || !pa_add_monomial(f, p, nn, d, e)) {
                fprintf(stderr, "--power: no field GF(%d^%d) of a usable size, or the terms have different N\n", p, nn);
                return 2;
            }
            n = nn;
            field = true;
        } else if (a[0] != '-') table = a;
        else {
            fprintf(stderr, "option %s is not available with --p %d\n", a.c_str(), p);
            return 2;
        }
    }
    if (f.empty()) {
        if (table.empty()) { usage(); return 2; }
        n = pa_parse(table, p, f);
        if (n < 1) { fprintf(stderr, "bad table: need exactly %d^n values between 0 and %d\n", p, p - 1); return 2; }
    }
    if (!fits(n)) { fprintf(stderr, "too large: at most 60 million points\n"); return 2; }
    std::vector<PMap> maps;
    if (known && field) {
        PGF K;
        if (K.init(p, n)) pa_field_maps(K, f, maps);
    } else if (known) fprintf(stderr, "note: --aut-known applies to --power functions only\n");
    if (scr) {
        f = pa_scramble(f, p, n, scr);
        pa_scramble_maps(p, n, scr, maps);
    }
    if (!aut_file.empty()) {
        std::vector<AffMap> am;
        if (!read_maps(aut_file.c_str(), n, am)) {
            fprintf(stderr, "cannot read the maps in %s\n", aut_file.c_str());
            return 2;
        }
        for (const AffMap& m : am) {
            PMap pm;
            pm.col = m.col;
            pm.t = m.t;
            maps.push_back(pm);
        }
    }
    PResult r = pa_solve(f, p, n, opt, target, mode, 3, maps.empty() ? nullptr : &maps);
    if (json) {
        pa_print_json(r, target);
        return 0;
    }
    if (target >= 0) {
        printf("p = %d, n = %d\nf is %d-normal: %s\n", p, n, target, r.k >= target ? "YES" : "NO");
        if (r.k < target) {
            printf("(best flat found so far has dimension %d; the maximum was not determined)\n", r.k);
            printf("time: %.3f s\n", r.seconds);
        }
    }
    if (target < 0 || r.k >= target) pa_report(r);
    if (target < 0 && r.bent) {
        if (n % 2 == 0) printf("f is bent and %s (n/2 = %d)\n", 2 * r.k == n ? "normal" : "NOT normal", n / 2);
        else printf("f is bent (in odd dimension a bent function is at most %d-normal)\n", (n - 1) / 2);
    }
    return 0;
}

// f + <u, x> for every u, the outer loop spread over the threads.
static void all_linear(const TT& F, int n, Options opt, int target) {
    double t0 = now_sec();
    int nt = opt.threads;
    opt.threads = 1;
    opt.verbose = false;
    opt.sym_cands = nullptr;
    opt.aut_source = nullptr;
    std::vector<u64> hist(n + 1, 0);
    std::vector<u32> example(n + 1, 0);
    std::atomic<u32> next(0);
    std::mutex mu;
    auto worker = [&]() {
        for (;;) {
            u32 u = next.fetch_add(1);
            if (u >= (1u << n)) break;
            Result r = solve(add_linear(F, n, u), n, opt, 0, target);
            std::lock_guard<std::mutex> g(mu);
            if (!hist[r.k]++ || u < example[r.k]) example[r.k] = u;
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < nt; ++i) th.emplace_back(worker);
    for (auto& t : th) t.join();
    printf("f + <u, x> over all %u linear functions (%.1f s):\n", 1u << n, now_sec() - t0);
    if (target >= 0) {
        u64 yes = 0;
        for (int k = target; k <= n; ++k) yes += hist[k];
        printf("  %d-normal for %llu choices of u, not %d-normal for %llu\n", target, (unsigned long long)yes, target,
               (unsigned long long)((1ull << n) - yes));
        return;
    }
    for (int k = 0; k <= n; ++k)
        if (hist[k]) printf("  k = %d for %llu choices of u (e.g. u = 0x%x)\n", k, (unsigned long long)hist[k], example[k]);
}

struct Case {
    std::string name;
    int n;
    std::function<Spec()> make;
};

// which: 1 = bent functions, 2 = other structured functions, 3 = both.
static void suite(const Options& opt0, bool big, int which) {
    std::vector<Case> cs;
    auto add = [&](std::string name, int n, std::function<Spec()> f) { cs.push_back({name, n, f}); };
    auto w3 = [](int n) { return ((1ull << n) - 1) / 3; };  // exponent of an element of order 3
    auto inv = [](int n) { return sp_field(power_function(n, (1ull << n) - 2, 0), n); };
    if (which & 1) {
        for (int n : {8, 10, 14, 16, 20}) add("Kasami Tr(a x^57)", n, [=] { return sp_field(kasami(n), n); });
        add("Kasami k=5, Tr(a x^993)", 14, [] { return sp_field(kasami(14, 5), 14); });
        add("Kasami k=5, Tr(a x^993)", 16, [] { return sp_field(kasami(16, 5), 16); });
        add("Kasami k=7, Tr(a x^16257)", 16, [] { return sp_field(kasami(16, 7), 16); });
        add("Kasami k=7, Tr(a x^16257)", 20, [] { return sp_field(kasami(20, 7), 20); });
        add("Kasami k=9, Tr(a x^261633)", 20, [] { return sp_field(kasami(20, 9), 20); });
        // nonnormal examples and their direct sums with hyperbolic pairs y1 y2 + ...
        add("dual of Kasami(10)", 10, [] { return sp_dual(sp_field(kasami(10), 10)); });
        add("dual of Kasami(14)", 14, [] { return sp_dual(sp_field(kasami(14), 14)); });
        add("Kasami(10) + x0", 10, [] { return sp_plus(sp_field(kasami(10), 10), 1); });
        for (int r : {1, 2, 3}) add("Kasami(10) + x0 + " + std::to_string(r) + " pairs", 10 + 2 * r,
                                    [=] { return sp_hyp(sp_plus(sp_field(kasami(10), 10), 1), r); });
        for (int r : {1, 2}) add("Kasami(14) + " + std::to_string(r) + " pairs", 14 + 2 * r,
                                 [=] { return sp_hyp(sp_field(kasami(14), 14), r); });
        for (int r : {1, 2}) add("same, random coordinates", 14 + 2 * r,
                                 [=] { return sp_scramble(sp_hyp(sp_field(kasami(14), 14), r), 5); });
        if (big) add("Kasami(14) + 3 pairs", 20, [] { return sp_hyp(sp_field(kasami(14), 14), 3); });
        for (int n : {8, 12, 16, 20}) add("Dillon Tr(a x^(2^(n/2)-1))", n, [=] { return sp_field(dillon(n), n); });
        for (int n : {12, 20}) {  // n = 4r, r odd: d = (2^r + 1)^2
            int r = n / 4;
            u64 d = (1ull << (2 * r)) + (1ull << (r + 1)) + 1;
            add("monomial Tr(a x^" + std::to_string(d) + ")", n, [=] { return sp_field(power_function(n, d, w3(n)), n); });
        }
        for (int n : {12, 16, 20}) add("Maiorana-McFarland", n, [=] { return sp_scramble(sp_plain(mm_bent(n, 1), n), 2); });
        for (int n : {12, 16}) add("Maiorana-McFarland class D0", n, [=] { return sp_scramble(sp_plain(mm_d0(n, 1), n), 2); });
        if (big) add("Maiorana-McFarland class D0", 20, [] { return sp_scramble(sp_plain(mm_d0(20, 1), 20), 2); });
        for (int n : {12, 16, 20}) add("partial spread PS_ap", n, [=] { return sp_scramble(sp_ps(ps_ap(n, 1), n), 2); });
    }
    if (which & 2) {
        for (int n : {8, 10, 12, 14, 16, 18, 20}) add("inverse Tr(x^-1)", n, [=] { return inv(n); });
        for (int n : {9, 11, 13, 15}) add("inverse Tr(x^-1)", n, [=] { return inv(n); });
        if (big) add("inverse Tr(x^-1)", 17, [=] { return inv(17); });
        add("AES S-box component", 8, [] { return sp_plain(aes_component(1), 8); });
        for (int n : {9, 11, 13, 15}) add("Kasami AB Tr(x^13)", n, [=] { return sp_field(power_function(n, 13, 0), n); });
        if (big) add("Kasami AB Tr(x^13)", 17, [] { return sp_field(power_function(17, 13, 0), 17); });
        for (int n : {9, 11, 13, 15}) {
            u64 d = (1ull << ((n - 1) / 2)) + 3;
            add("Welch AB Tr(x^" + std::to_string(d) + ")", n, [=] { return sp_field(power_function(n, d, 0), n); });
        }
        for (int n : {10, 12, 14}) add("Carlet-Feng", n, [=] { return sp_field(carlet_feng(n), n); });
        if (big) add("Carlet-Feng", 16, [] { return sp_field(carlet_feng(16), 16); });
        for (int n : {9, 13, 15}) add("majority", n, [=] { return sp_symmetric(threshold_function(n, n / 2 + 1), n); });
        if (big) add("majority", 17, [] { return sp_symmetric(threshold_function(17, 9), 17); });
        for (int n : {8, 12, 16, 18}) add("hidden weighted bit", n, [=] { return sp_plain(hwbf(n), n); });
        if (big) add("hidden weighted bit", 20, [] { return sp_plain(hwbf(20), 20); });
    }
    printf("%-30s %3s %9s %3s %5s %9s %7s %8s %9s\n", "function", "n", "max|W|", "k", "on f=", "bounds", "greedy", "searches", "time");
    for (const Case& c : cs) {
        Spec sp = c.make();
        if (sp.F.empty()) { printf("%-30s %3d  (not available)\n", c.name.c_str(), c.n); continue; }
        long long mw = max_walsh(sp.F, c.n);
        bool bent = mw * mw == (1ll << c.n);
        Options opt = opt0;
        opt.sym_cands = sp.sym.empty() ? nullptr : &sp.sym;
        Result r = solve(sp.F, c.n, opt);
        char bd[32], sy[64] = "";
        snprintf(bd, sizeof bd, "%d/%d", r.ub[0], r.ub[1]);
        if (r.sym_loaded && (r.sym_gens || r.sym_flip)) {
            if (r.sym_order >= 1) snprintf(sy, sizeof sy, "  |G| = %.0f%s", r.sym_order, r.sym_flip ? ", f~f+1" : "");
            else snprintf(sy, sizeof sy, "  |G| > %zu%s", SYM_ROOT_CAP, r.sym_flip ? ", f~f+1" : "");
        }
        printf("%-30s %3d %9lld %3d %5d %9s %7d %8d %8.2fs%s%s\n", c.name.c_str(), c.n, mw, r.k, r.value, bd, r.greedy_k,
               r.searches, r.seconds, !bent ? "" : 2 * r.k == c.n ? "  bent, normal" : "  bent, NONNORMAL", sy);
        fflush(stdout);
    }
}

// Candidate maps from a file: per line the images of e_0 .. e_(n-1) and then A(0).
static bool read_maps(const char* path, int n, std::vector<AffMap>& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        size_t h = line.find('#');
        if (h != std::string::npos) line.resize(h);
        for (char& ch : line)
            if (ch == ',' || ch == ';') ch = ' ';
        std::istringstream ls(line);
        std::vector<u32> v;
        std::string tok;
        while (ls >> tok) v.push_back((u32)strtoul(tok.c_str(), nullptr, 0));
        if (v.empty()) continue;
        if ((int)v.size() != n && (int)v.size() != n + 1) {
            fprintf(stderr, "%s: a line has %zu numbers, expected %d or %d\n", path, v.size(), n, n + 1);
            return false;
        }
        AffMap a;
        a.col.assign(v.begin(), v.begin() + n);
        a.t = (int)v.size() == n + 1 ? v[n] : 0;
        out.push_back(a);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc == 5 && !strcmp(argv[1], "--aut-worker")) return aut_worker(argv[2], argv[3], atof(argv[4]));
    aut_set_self(argv[0]);
    for (int i = 1; i + 1 < argc; ++i)
        if (!strcmp(argv[i], "--p") && atoi(argv[i + 1]) != 2) return pary_main(argc, argv, atoi(argv[i + 1]));
    Options opt;
    opt.threads = std::max(1u, std::thread::hardware_concurrency());
    bool hex = false, alllin = false, big = false, dual = false, json = false;
    int run_suite = 0;
    u32 plus = 0;
    int target = -1, hyp = 0;
    u64 scr = 0;
    std::string table, aut_file;
    for (int i = 1; i < argc; ++i)
        if (!strcmp(argv[i], "--aut-known")) g_known_sym = true;
    Spec sp;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto num = [&](int j) { return j < argc ? strtoull(argv[j], nullptr, 0) : 0ULL; };
        auto has = [&](int j) { return j < argc && isdigit((unsigned char)argv[j][0]); };
        if (a == "--selftest") return selftest() ? 1 : 0;
        else if (a == "--selftest-p") {  // the tests of the p-ary solver only
            int checks = 0;
            const int fails = pary_selftest(checks);
            printf("selftest: %d checks, %d failures\n", checks, fails);
            return fails ? 1 : 0;
        }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--suite") run_suite = 3;
        else if (a == "--suite-bent") run_suite = 1;
        else if (a == "--suite-other") run_suite = 2;
        else if (a == "--big") big = true;
        else if (a == "--p" && has(i + 1)) ++i;  // --p 2: the binary solver
        else if (a == "--hex") hex = true;
        else if (a == "--json") json = true;
        else if (a == "--all-linear") alllin = true;
        else if (a == "--dual") dual = true;
        else if (a == "--plus" && has(i + 1)) plus = (u32)num(++i);
        else if (a == "--hyp" && has(i + 1)) hyp = (int)num(++i);
        else if (a == "--scramble" && has(i + 1)) scr = num(++i);
        else if (a == "--target" && has(i + 1)) target = (int)num(++i);
        else if (a == "-v") opt.verbose = true;
        else if (a == "--no-sparse") opt.use_sparse = false;
        else if (a == "--no-shadow") opt.use_shadow = false;
        else if (a == "--no-prefix") opt.use_prefix = false;
        else if (a == "--no-quick") opt.use_quick = false;
        else if (a == "--aut-known") {}
        else if (a == "--sym-given") opt.sym_coords = false;
        else if (a == "--no-mates") opt.sym_mates = false;
        else if (a == "--aut") {
            opt.aut_budget = has(i + 1) ? atof(argv[++i]) : 60.0;
            opt.aut_source = aut_library_start;
        } else if (a == "--aut-file" && i + 1 < argc) aut_file = argv[++i];
        else if (a == "-t" && has(i + 1)) opt.threads = std::max(1, (int)num(++i));
        else if (a == "--sparse-weight" && i + 1 < argc) opt.sparse_weight = atof(argv[++i]);
        else if (a == "-f" && i + 1 < argc) {
            std::ifstream in(argv[++i]);
            if (!in) { fprintf(stderr, "cannot open %s\n", argv[i]); return 2; }
            std::stringstream ss;
            ss << in.rdbuf();
            table = ss.str();
        } else if (a == "--random" && has(i + 1)) {
            int n = (int)num(++i);
            sp = sp_plain(random_function(n, has(i + 1) ? num(++i) : 1), n);
        } else if (a == "--mm" && has(i + 1)) {
            int n = (int)num(++i);
            u64 seed = has(i + 1) ? num(++i) : 1;
            sp = sp_scramble(sp_plain(mm_bent(n, seed), n), seed + 1000);
        } else if (a == "--ps" && has(i + 1)) {
            int n = (int)num(++i);
            u64 seed = has(i + 1) ? num(++i) : 1;
            sp = sp_scramble(sp_ps(ps_ap(n, seed), n), seed + 1000);
        } else if (a == "--kasami" && has(i + 1)) {
            int n = (int)num(++i);
            sp = sp_field(kasami(n, has(i + 1) ? (int)num(++i) : 3), n);
        } else if (a == "--dillon" && has(i + 1)) {
            int n = (int)num(++i);
            sp = sp_field(dillon(n), n);
        } else if (a == "--d0" && has(i + 1)) {
            int n = (int)num(++i);
            u64 seed = has(i + 1) ? num(++i) : 1;
            sp = sp_scramble(sp_plain(mm_d0(n, seed), n), seed + 1000);
        } else if (a == "--power" && has(i + 2)) {
            int n = (int)num(++i);
            u64 d = num(++i);
            sp = sp_field(power_function(n, d, has(i + 1) ? num(++i) : 0), n);
        } else if (a == "--inverse" && has(i + 1)) {
            int n = (int)num(++i);
            sp = sp_field(power_function(n, (1ull << n) - 2, 0), n);
        } else if (a == "--aes" && has(i + 1)) {
            sp = sp_plain(aes_component((u32)num(++i)), 8);
        } else if (a == "--cf" && has(i + 1)) {
            int n = (int)num(++i);
            sp = sp_field(carlet_feng(n), n);
        } else if (a == "--maj" && has(i + 1)) {
            int n = (int)num(++i);
            sp = sp_symmetric(threshold_function(n, n / 2 + 1), n);
        } else if (a == "--threshold" && has(i + 2)) {
            int n = (int)num(++i);
            sp = sp_symmetric(threshold_function(n, (int)num(++i)), n);
        } else if (a == "--hwb" && has(i + 1)) {
            int n = (int)num(++i);
            sp = sp_plain(hwbf(n), n);
        } else if (a[0] != '-') table = a;
        else { usage(); return 2; }
    }
    if (opt.aut_source && !aut_library_built())
        fprintf(stderr, "note: this binary was built without the equivalence library, --aut has no effect (see README.md)\n");
    if (run_suite) { suite(opt, big, run_suite); return 0; }
    if (sp.n < 0) {
        if (table.empty()) { usage(); return 2; }
        sp.n = hex ? parse_hex(table, sp.F) : parse_binary(table, sp.F);
        if (sp.n < 0) { fprintf(stderr, "bad truth table: need exactly 2^n values\n"); return 2; }
    }
    if (sp.n < 1 || sp.n > 30 || sp.F.size() != nwords(sp.n)) {
        fprintf(stderr, "n must be between 1 and 30 (field-based families: up to 20)\n");
        return 2;
    }
    if (dual) {
        sp = sp_dual(sp);
        if (sp.F.empty()) { fprintf(stderr, "--dual needs a bent function\n"); return 2; }
    }
    if (plus) sp = sp_plus(sp, plus);
    if (hyp) sp = sp_hyp(sp, hyp);
    if (scr) sp = sp_scramble(sp, scr);
    const int n = sp.n;
    if (!aut_file.empty() && !read_maps(aut_file.c_str(), n, sp.sym)) {
        fprintf(stderr, "cannot read the maps in %s\n", aut_file.c_str());
        return 2;
    }
    if (!sp.sym.empty()) opt.sym_cands = &sp.sym;
    if (opt.verbose) fprintf(stderr, "max |Walsh| = %lld\n", (long long)max_walsh(sp.F, n));
    if (alllin) { all_linear(sp.F, n, opt, target); return 0; }
    Result r = solve(sp.F, n, opt, 3, target);
    if (json) {
        print_json(r, n, target, sp.F);
        return r.k < 0 ? 1 : 0;
    }
    if (target >= 0) {
        printf("n = %d\nf is %d-normal: %s\n", n, target, r.k >= target ? "YES" : "NO");
        if (r.k < target) {
            printf("(best flat found so far has dimension %d; the maximum was not determined)\n", r.k);
            print_search_stats(r);
            print_symmetry(r);
            printf("time: %.3f s\n", r.seconds);
        }
    }
    if (target < 0 || r.k >= target) report(r);
    if (target < 0 && n % 2 == 0) {
        long long mw = max_walsh(sp.F, n);
        if (mw * mw == (1ll << n)) printf("f is bent and %s (n/2 = %d)\n", 2 * r.k == n ? "normal" : "NOT normal", n / 2);
    }
    return r.k < 0 ? 1 : 0;
}
