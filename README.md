# normality

Computes the largest `k` such that a Boolean function `f : F_2^n -> F_2` is
**k-normal**, i.e. constant on some `k`-dimensional affine subspace (flat), and
prints a flat that attains it. For a bent function this answers "is it
normal?" (`k = n/2`) and, if not, by how much it misses.

The answer is exact. The flat is re-checked point by point against the truth
table, and optimality is established either by a spectral bound or by an
exhaustive search showing that no `(k+1)`-flat exists.

The exhaustive search can use the automorphism group of `f` to look at one
flat per orbit only. The group can come from an external equivalence library,
from the construction of a built-in family, or from a file; every map is
verified against the truth table before it is used.

## Python

```python
import normality

table = "0110100110010110"            # truth table: character i is f(i); bit j of i is x_j
r = normality.solve(table)
r.k                                   # 3: the largest k such that f is k-normal
r.value, r.offset, r.basis            # f == 0 on the flat 0 + span(3, 5, 9)
list(r.points())                      # the 2^k points of that flat
normality.check(table, r)             # verify the flat in Python

# other ways to give the function
normality.solve([0, 1, 1, 0, 1, 0, 0, 1])                 # list, tuple, bytes, numpy array of 0/1
normality.solve(lambda x: bin(x).count("1") >= 5, n=9)    # a Python function of the point x
normality.solve(0x6996, n=4)                              # an integer whose bit i is f(i)

normality.largest_k(table)            # just the number
normality.is_k_normal(table, 3)       # only this k: much cheaper than the maximum
normality.is_normal(table)            # k = ceil(n/2)

r = normality.family("kasami", 14)    # a built-in family (normality.FAMILIES lists them)
r.bent, r.normal, r.seconds           # True, False, 0.03
normality.family("kasami", 14, hyp=2, known_symmetry=True)   # 18 variables, with its known symmetries
```

`normality.py` sits next to the C++ sources and needs nothing but the Python
standard library. Import it from this folder (or put the folder on
`PYTHONPATH`). The solver is compiled on first use, which needs `g++`;
`normality.build()` does the same explicitly, and
`python -m unittest test_normality` runs the interface tests.

`solve` also takes `target=k`, `threads=`, `verbose=True`, and the symmetry
options described below: `aut=True` (the equivalence library, if it was built
in) and `maps=[...]` (automorphisms you know, each as the images of
`e_0 .. e_(n-1)` followed by the image of 0). The result says whether `k` is
exact (`r.exact`; it is not when a `target` cut the search short), gives the
spectral upper bounds, and reports what the symmetry contributed
(`r.symmetry`).

## Build and run

Without the equivalence library (everything except `--aut` works):

```bash
g++ -O3 -march=native -std=c++17 -pthread -o normality main.cpp
```

With the library
([zskiley/Affine-and-linear-equivalence-algorithms](https://github.com/zskiley/Affine-and-linear-equivalence-algorithms),
C++20), cloned next to the sources:

```bash
git clone https://github.com/zskiley/Affine-and-linear-equivalence-algorithms third_party/aleq
```

```bash
g++ -O3 -march=native -std=c++20 -pthread -DNORMALITY_ALEQ -I third_party/aleq/src -o normality main.cpp autgroup.cpp third_party/aleq/src/dfs.cpp third_party/aleq/src/equivalence_api.cpp
```

With MinGW-w64 on Windows add `-static -Wa,-muse-unaligned-vector-move` to
either command. The second flag works around a GCC/Windows stack-alignment
problem with AVX (without it the program can crash); `-static` avoids loading
an older C++ runtime DLL that happens to be earlier on the `PATH`.

The library was used at commit `7b97dc0` (23 July 2026). That repository has
no licence file, so none of its code is included here; `third_party/aleq` is
only a local clone.

```bash
./normality 0110100110010110            # truth table, character i = f(i)
./normality --hex 6996                  # same table as a hex integer (bit i = f(i))
./normality -f table.txt                # table from a file
./normality --kasami 14 -v              # built-in family, with progress
./normality --kasami 14 --target 7      # only decide "is f 7-normal?"
./normality --kasami 14 --dual          # the dual bent function
./normality --kasami 14 --hyp 2         # f(x) + y1 y2 + y3 y4, a bent function in 18 variables
./normality --kasami 10 --all-linear    # k for f + <u,x>, every u
./normality --inverse 15 --aut          # compute the automorphism group with the library and use it
./normality --kasami 14 --hyp 3 --aut-known   # use the symmetries known from the construction
./normality -f table.txt --aut-file maps.txt  # use the maps in a file (n+1 numbers per line)
./normality --suite-bent                # bent benchmark set (--suite: everything, --big: slow cases too)
./normality --selftest                  # correctness tests
```

Variable `x_j` is bit `j` of the index `i`. `-t N` sets the thread count
(default: all cores). `--help` lists the built-in families and modifiers.
AVX2 and BMI2 are used when the compiler enables them; the code also builds
without.

## Algorithm

Write `S_c = f^{-1}(c)`. The task is to find the largest flat inside `S_0` or
`S_1`.

1. **Linear structures.** If `f(x+v) = f(x)` for all `x`, every maximal flat
   contains the direction `v`. These `v` form a subspace `L`, read off the
   autocorrelation; the problem is solved on `F_2^n / L` and `dim L` is added.

2. **Spectral upper bound, one per constant.** If `f = c` on `a + E` with
   `dim E = k`, then

   `sum over u in E^perp, u != 0, of (-1)^(a.u) W_f(u) = +-2 |S_(1-c)|`,

   so the `2^(n-k) - 1` largest `|W_f(u)|`, `u != 0`, must add up to at least
   `2 |S_(1-c)|`. This sharpens the classical `2^k <= max |W_f|`, and for a
   bent function it leaves only one constant that can be attained on an
   `n/2`-flat.

3. **Lower bounds.** (a) Greedy: repeatedly add the direction that keeps the
   most cosets inside `S_c` (autocorrelation by fast Walsh transform), with a
   few randomised restarts. (b) A search restricted to flats through the
   origin, i.e. linear subspaces. A direction is then admissible only if it
   lies in the current set, which makes the search far smaller; it runs on a
   fixed work budget because it only supplies a lower bound.

4. **Exhaustive search for a `t`-flat**, for `t` = upper bound, upper bound
   - 1, ... until one succeeds or `t` reaches the lower bound. Refuting a large
   `t` is cheap and the cost rises steeply as `t` falls, so this order never
   searches below the true maximum. When both constants are possible they are
   searched at the same time, so a quick success for one does not wait for a
   slow refutation of the other.

The exhaustive search (`search.hpp`) is where the work is:

- **Quotient bitsets.** A set `T` in `F_2^m` is a bitset of `2^m` bits. After
  choosing a direction `v` with leading bit `p`, pass to `F_2^m / <v>`: delete
  coordinate `p` and keep the cosets `{x, x+v}` that lie inside `T`. This is
  one AND of the bitset with an XOR-translate of itself. It handles every
  translate of the partial direction space at once, and the set halves at each
  level. The leaf of a successful path is the set of all flats with that
  direction space.
- **No duplicates.** Directions are chosen as the reduced echelon basis of `E`
  in order of increasing leading bit, so each direction space is visited once.
- **Prefix restriction.** A `j`-dimensional space meets every subspace of
  dimension `m-j+1`, so the next (lowest) basis vector has leading bit
  `<= m-j`: `2^(m-j+1)` candidates instead of `2^m`.
- **Shadow tests.** The remaining directions have leading bits `>= lo`, so
  dropping the low `lo` coordinates is injective on the flat, and the set of
  non-empty `2^lo`-blocks must itself contain a flat of the same dimension.
  Projecting further, onto the top six coordinates (64 "buckets"), the flat
  becomes an `i`-flat of buckets and meets `2^(j-i)` blocks in each of them;
  so for some `i` the buckets holding that many non-empty blocks must contain
  an `i`-flat. Both conditions are decided on one 64-bit word.
- **Two-stage candidates.** A candidate direction is first evaluated without
  storing the quotient set, only the per-bucket block counts the shadow tests
  need. Survivors get a quick look at a few well-spread buckets: either one
  pair difference is common to all of them, or the buckets met form a
  hyperplane and each holds a repeated difference. Only what survives that is
  materialised and searched.
- **Sparse leaves.** Small sets are held as sorted point lists; viable
  directions are found by counting pair differences, bucket by bucket, and a
  class of directions is dropped as soon as it cannot reach its bucket quota.
- **Coordinates.** Normality does not depend on coordinates but the pruning
  does (structure aligned with the given coordinates can help or hurt by a
  factor of several). For long searches the same sample of root candidates is
  raced in the given coordinates and in a random re-coordinatisation, and the
  full search runs in the winner.
- **SIMD and threads.** The candidate loop is AVX2, with the low six bits of
  `v` enumerated in Gray-code order so the in-word permutation is one swap per
  step; root candidates are distributed over threads.

Cost is governed by how many direction spaces of small dimension must be
examined before the sets become too small, roughly `2^(d (n-t))` at depth `d`
when refuting a `t`-flat. So refuting flats of dimension near `n/2` (bent
functions) is comparatively cheap, and refuting flats of dimension near
`log2 n` in a large random-like function is not.

### Symmetry

Let `G` be a group of affine maps `A(x) = Mx + b` with `f(Ax) = f(x)`. Such a
map takes flats in `S_c` to flats in `S_c`, and the direction space `E` of a
flat to `M E`. So it is enough to look at one direction space per orbit of the
linear parts `M` (`group.hpp`).

The search already writes every `E` in one way: as the sequence
`(w_1, w_2, ...)` of its echelon basis, each vector in the quotient
coordinates of its level. Here `w_1` is the smallest non-zero vector of `E`
(as an integer), `w_2` the smallest non-zero vector of `E / <w_1>`, and so on.
Among the images of `E` under the group, take the one whose sequence is
lexicographically smallest. For that one:

- **No vector of `E` lies in an orbit that contains a vector below `w_1`.**
  So `w_1` is the smallest vector of its own orbit: only one root direction
  per orbit is tried. And below a root, a direction is skipped if its coset
  modulo the directions chosen so far contains a vector of `low(w_1)`, the
  union of the orbits with a smaller minimum. That set of cosets is carried
  down the tree as one more bitset per level, updated by the OR-analogue of
  the quotient step. It needs only the orbits, so it works for groups far too
  large to list.
- **The same holds one level down with the stabiliser of `w_1`** acting on the
  quotient, and so on down the chain of stabilisers: a candidate is skipped if
  the stabiliser of the directions chosen so far maps it to a smaller vector.
  Stabilisers come from the list of all group elements when the group has at
  most 4096 of them, and from random Schreier elements otherwise. Using only
  part of a stabiliser is still sound; it prunes less. For heavy subtrees the
  orbits of the stabiliser are computed on the whole quotient, which gives the
  forbidden-coset rule at depths 1 and 2 as well.

- **Other images that start with the same vector.** If the partial space
  holds another vector `u` from the orbit of `w_1`, the group element taking
  `u` to `w_1` gives another image of the space that also starts with `w_1`,
  and its second direction must not be smaller than `w_2`. Each such `u` is
  checked against the vectors already there, and the vectors it rules out are
  added to the forbidden cosets. With this the second direction is compared
  over the whole group, not only over the stabiliser (`--no-mates` switches it
  off). It matters when orbits are a large part of the space: for majority at
  `n = 17` it halves the number of nodes. On the bent direct sums it changes
  nothing measurable, because those searches die at depth 4 or 5, where the
  partial space has at most 31 vectors and almost never holds a second vector
  of that orbit (0.4% of the extensions at `n = 18`).

Three more things follow from the group:

- **One constant instead of two.** If some affine map takes `f` to `f + 1`,
  flats in `S_0` and `S_1` correspond, and only one constant is searched.
- **Coordinates adapted to the group.** How much is pruned depends on how the
  orbits lie relative to the low coordinates, where the first directions are
  taken from. A basis is chosen greedily so that the span of its first vectors
  meets as few orbits as possible; the rest of the basis is random. Long
  searches run in these coordinates.
- **Splitting heavy subtrees.** With symmetry most of the work sits under a
  few root directions, and one subtree is one thread's job. So every root
  first gets a work allowance; roots that exceed it are searched again, split
  among all threads by their second direction, and second directions that
  exceed the allowance are split by the third.

Where the maps come from:

- `--aut [SEC]`: the external library computes generators of the full group of
  affine self-equivalences. It cannot be interrupted, so it runs in a child
  process (this executable started as a worker), beside the searches, for at
  most `SEC` seconds (default 60). Its answer is picked up when it arrives; a
  search that is running at that moment starts over with the group. If the
  searches finish first, the worker is stopped.
- `--aut-known`: the symmetries known from the construction of a built-in
  family: maps `x -> c x^(2^i)` of the field for trace functions, isometries
  of the hyperbolic part for `--hyp`, coordinate permutations for symmetric
  functions, and their images under `--scramble` and `--dual`.
- `--aut-file FILE`: one map per line, the images of `e_0 .. e_(n-1)` followed
  by the image of 0.

Nothing depends on these being right. Each map is checked on all `2^n` points
(`f(Ax) = f(x)` or `f(Ax) = f(x) + 1` everywhere) and dropped otherwise, and
the pruning is sound for any set of verified maps, whether or not they
generate the whole automorphism group.

## Results on bent functions

Ryzen 7 7735HS, 16 threads (`./normality --suite-bent`, and the same with
`--aut-known`). Every function in the table was checked to be bent from its
Walsh spectrum. `k` is the largest dimension of a flat on which `f` is
constant, so the function is normal iff `k = n/2`. The last two columns are
the total time without symmetry and with the symmetries of the construction;
`|G|` is the order of the group of linear parts in use. Timings on this
laptop vary by some tens of percent between runs.

| function | n | k | normal | time | with symmetry |
| --- | ---: | ---: | :---: | ---: | ---: |
| Kasami `Tr(a x^57)`, `a` of order 3 | 8, 10 | 4, 5 | yes | 0.00 s | |
| | 14 | 6 | **no** | 0.03 s | 0.03 s, `\|G\|` = 21 |
| | 16 | 8 | yes | 0.01 s | |
| | 20 | 10 | yes | 0.27 s | |
| Kasami `Tr(a x^(4^k-2^k+1))`, k = 5 | 14 | 6 | **no** | 0.03 s | 0.03 s, `\|G\|` = 21 |
| k = 5, 7, 9 | 16, 20 | 8, 10 | yes | < 0.2 s | |
| dual of Kasami(10) | 10 | 5 | yes | 0.00 s | |
| dual of Kasami(14) | 14 | 6 | **no** | 0.03 s | 0.03 s, `\|G\|` = 21 |
| Kasami(10) + `x0` | 10 | 4 | **no** | 0.00 s | no symmetry left |
| Kasami(10) + `x0` + 1, 2 hyperbolic pairs | 12, 14 | 5, 6 | **no** | < 0.1 s | < 0.1 s, `\|G\|` = 2, 72 |
| Kasami(10) + `x0` + 3 hyperbolic pairs | 16 | 7 | **no** | 0.72 s | 0.23 s, `\|G\|` = 40320 |
| Kasami(14) + 1 hyperbolic pair | 16 | 7 | **no** | 0.53 s | 0.34 s, `\|G\|` = 42 |
| the same after a random affine change of variables | 16 | 7 | **no** | 1.5 s | 1.2 s |
| Kasami(14) + 2 hyperbolic pairs | 18 | 8 | **no** | 10.3 s | 2.3 s, `\|G\|` = 1512 |
| the same after a random affine change of variables | 18 | 8 | **no** | 10.1 s | 3.2 s |
| Kasami(14) + 3 hyperbolic pairs | 20 | 9 | **no** | 380 s (*) | 3.8 s, `\|G\|` = 21 x 40320 |
| Kasami(14) + 4 hyperbolic pairs | 22 | < 11 | **no** | not run | 12.6 s (**) |
| Dillon `Tr(a x^(2^(n/2)-1))`, `a` a Kloosterman zero | 8, 12, 16, 20 | n/2 | yes | < 0.1 s | |
| monomial `Tr(a x^((2^r+1)^2))`, n = 4r | 12, 20 | n/2 | yes | < 0.1 s | |
| Maiorana-McFarland (scrambled) | 12, 16, 20 | n/2 | yes | < 0.1 s | |
| Maiorana-McFarland class D0 (scrambled) | 12, 16 | n/2 | yes | 0.43 s | none known |
| | 20 | 10 | yes | 182 s | none known |
| partial spread PS_ap (scrambled) | 12, 16 | n/2 | yes | < 0.1 s | |
| | 20 | 10 | yes | 1.3 s | 1.2 s, `\|G\|` = 1023 |

(*) Without symmetry only the proof that no 10-flat exists was run
(`--target 10`, 380 s). With symmetry that proof takes 2.5 s and the full
answer `k = 9` takes 3.8 s.

(**) `--target 11`: the proof that no 11-flat exists. The maximum was not
computed for this one.

An empty last column means no exhaustive search was needed, so the symmetry
was never looked at. The program lists a group only up to order 4096; the
larger orders in the table are those of the orthogonal group of the
hyperbolic part (40320 for three pairs, as listed by an earlier build with a
higher limit) times the 21 of the Kasami function.

Notes:

- **Comparison with the literature** (Canteaut, Daum, Dobbertin, Leander,
  *Finding nonnormal bent functions*, Discrete Applied Mathematics 154, 2006).
  For `n = 10` the Kasami function itself is 5-normal, but `f + <u,x>` is
  nonnormal for 870 of the 1024 choices of `u`
  (`--kasami 10 --all-linear`, 0.1 s); the paper's nonnormal 10-variable bent
  function is such a modified Kasami function. For `n = 14` the function is
  not constant on any 7-flat, and neither is `f + <u,x>` for any of the 16384
  choices of `u` (`--kasami 14 --all-linear --target 7`, 6.5 minutes), i.e. it is
  not weakly normal, which is the paper's result for this function.
- **Kasami functions.** Among the cases computed, the Kasami bent functions
  are nonnormal exactly for `n = 14` (both `k = 3` and `k = 5`), and the dual
  of the 14-variable one is nonnormal too. For `n = 8, 10, 16, 20` a flat of
  dimension `n/2` is found at once (for 16 and 20 a subspace through the
  origin).
- **Direct sums.** Adding hyperbolic pairs `y1 y2 + y3 y4 + ...` to a
  nonnormal example kept it nonnormal in every case computed, with
  `k = n/2 - 1` wherever the maximum was computed (up to 20 variables). These
  are the instances on which refuting normality is expensive, and where the
  symmetry pays most: the orthogonal group of the quadratic part acts on
  them, and the 20-variable refutation goes from 380 s to 2.5 s.
- **Hard to find rather than hard to refute.** The scrambled class-D0
  function in 20 variables is normal but the search needs 3 minutes to reach
  a 10-flat: neither lower-bound pass finds one, and the search has to go
  through a good part of the space first. No symmetry is supplied for it.

## Results on other structured functions

`bounds` are the spectral upper bounds for `f = 0 / f = 1`. Times are without
symmetry (`--suite-other`), with the symmetries of the construction
(`--aut-known`), and with the group computed by the library while the search
runs (`--aut`).

| function | n | k | bounds | time | `--aut-known` | `--aut` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inverse `Tr(x^-1)`, n even | 8 ... 20 | n/2 | n/2 | < 0.3 s | | |
| inverse `Tr(x^-1)`, n odd | 9, 11, 13 | 4, 4, 5 | 5, 6, 7 | < 0.1 s | | |
| | 15 | 5 | 8/8 | 12.3 s | 1.1 s | 2.6 s |
| | 17 | 6 | 9/9 | (a) | 209 s | not run |
| AES S-box component | 8 | 4 | 4/4 | 0.00 s | | |
| Kasami AB `Tr(x^13)` | 9, 11, 13 | 4, 5, 6 | 4, 5, 6 | 0.01 s | | |
| | 15 | 6 | 7/7 | 0.84 s | 0.24 s | 1.0 s (b) |
| | 17 | 7 | 8/8 | (a) | 25.6 s | not run |
| Welch AB `Tr(x^(2^((n-1)/2)+3))` | 9, 11, 13 | 4, 5, 6 | 4, 5, 6 | 0.01 s | | |
| | 15 | 6 | 7/7 | 0.74 s | 0.29 s | 0.9 s (b) |
| | 17 | 7 | 8/8 | (a) | 24.3 s | not run |
| Carlet-Feng | 10, 12, 14 | 4, 5, 5 | 5, 6, 8 | < 0.7 s | trivial group | trivial group |
| majority | 9, 13, 15 | 4, 6, 7 | 6, 10, 12 | < 0.5 s | < 0.4 s | as without (b) |
| | 17 | 8 | 14/14 | 47.7 s | 0.43 s | 57.8 s (b) |
| hidden weighted bit | 8, 12, 16 | 4, 7, 10 | 5, 9, 13 | 0.1 s | none known | 0.1 s |
| | 18 | 11 | 15/15 | 3.2 s | none known | 2.4 s |
| | 20 | 12 | 17/17 | 59 s (c) | none known | 18.2 s |

(a) Did not finish in 2 minutes without symmetry (earlier build).

(b) The library needs longer than the search itself (about 10 s for the AB
functions at `n = 15`, more than a minute for majority at `n = 17`), so the
search finishes without its answer. For majority at `n = 17` the worker also
takes a core from the search.

(c) Measured with an earlier build; not re-run.

What the symmetry is in each case: the Frobenius map for the trace functions
(order `n`); the symmetric group `S_(n+1)` together with complementation for
majority; and for the hidden weighted bit function only a map taking `f` to
`f + 1`, which the library finds and which halves the work.

## Limits

- **The library and bent functions.** On its own the library took 15 s for
  the 10-variable Kasami function and did not finish within 90 s for any bent
  function tried in 12 or more variables (Dillon, partial spread and
  Maiorana-McFarland at `n = 12`, Kasami at `n = 14`). On non-bent functions
  it is fast (inverse at `n = 16`: 2 s; majority at `n = 13`, a group of order
  `14! x 2`: 1 s). So for bent functions the group has to come from
  `--aut-known` or `--aut-file`.
- **Symmetry helps only as much as the group is large and the search long.**
  The 14-variable Kasami function has a group of order 21 and a search of
  0.03 s: nothing to gain. A trivial group (Carlet-Feng, random functions)
  gives nothing.
- Refuting normality of a bent direct sum without symmetry gets roughly 15 to
  50 times dearer with every two extra variables (about 0.5 s, 10 s and 6
  minutes for `n` = 16, 18, 20). With the symmetries of the construction it
  takes about 0.3 s, 0.9 s, 2.5 s and 13 s for `n` = 16, 18, 20, 22.
- Outside the bent case, the hard instances are large functions whose
  normality is far below the spectral bound and close to that of a random
  function. Carlet-Feng at `n = 16` did not finish in 9 minutes (earlier
  build; its group is trivial), and the inverse function at `n = 17` takes
  3.5 minutes even with its group of order 17.
- **Tried and not adopted: the spectral bound inside the search.** The bound
  of step 2 also holds for every quotient set (a `j`-flat in `T` forces the
  `2^(m-j) - 1` largest Walsh coefficients of `T` to add up to `2^m - |T|`).
  Evaluated at every dense node without pruning, it failed for 9,371 of
  520,475 depth-2 nodes of a random 14-variable function, with 0.2% of the
  search time under them; 0.6% for the inverse function at `n = 13`, and no
  node at all for the Kasami bent function, its direct sum, or Kasami AB at
  `n = 15`. The transform would add 55 to 93% to the run time. The sets
  shrink so fast from one level to the next that a node which passes the
  counting test almost always passes this one too.
- Memory is a few bitsets of `2^n` bits per thread plus two integer arrays of
  `2^n` entries for the spectral step, and with symmetry a few more arrays of
  `2^n` 32-bit entries, so `n` up to the low twenties is practical. The
  field-based built-in families go up to `n = 20`.
- `--all-linear` runs `2^n` independent solves, without symmetry. Weak
  normality (the largest flat on which `f` is affine) could be decided by one
  search with two coupled bitsets instead; that is not implemented.

## Correctness checks

`--selftest` (about a minute and a half, 44,769 checks) covers:

- an independent brute force over all flats for `n <= 7`, under all eight
  combinations of the sparse / shadow / prefix prunings;
- the origin-anchored search against a brute force over all subspaces;
- closed forms for quadratic and Maiorana-McFarland functions;
- agreement between pruning configurations and thread counts for `n = 8..14`;
- flats planted in random functions (`n = 10..16`) that must be found, with
  the quick check on and off;
- 4,000 flats planted inside single search nodes with prescribed leading bits: no
  pruning test may reject such a node and the node search must succeed;
- the bit-level and SIMD helpers against their definitions;
- symmetry: stabilisers, skip sets, orbit minima and the group elements read
  off the orbit trees against the full list of group elements for random
  small groups;
- symmetry: random sets invariant under a random group of affine maps, with a
  planted flat. The search with symmetry must give the same answer as without
  for the planted dimension and the next one, with the full element list and
  with generators only, in given and in adapted coordinates, with the
  second-direction comparison over the whole group on and off, and with a
  tiny work allowance that forces every root and every second direction
  through the split passes;
- symmetry: the whole solver with known symmetries against itself without
  them (and against the brute force for `n <= 7`), including a map taking `f`
  to `f + 1` and maps that are not automorphisms and must be rejected;
- a group that arrives while the searches are under way, at once, after a few
  polls, or never;
- the library, if built in: every map it returns is an automorphism, answers
  do not change, and its time limit is enforced.

Both benchmark suites give the same `k` for all 64 functions with and without
symmetry.

## Files

| file | content |
| --- | --- |
| `bitops.hpp` | bitset primitives, quotient step, one-word flat test, shadow tests |
| `group.hpp` | linear maps, orbits, stabilisers, coordinates adapted to a group |
| `search.hpp` | exhaustive search (dense and sparse nodes, quick check, symmetry) |
| `solver.hpp` | linear structures, bounds, lower-bound passes, top-down driver |
| `gen.hpp` | truth-table parsing, function families and their known symmetries |
| `autlib.hpp` | runs the equivalence library in a child process |
| `autgroup.cpp` | the one file that includes the library's headers (optional) |
| `tests.hpp` | self-test and brute force |
| `main.cpp` | command line |
| `normality.py` | Python interface (runs the solver, reads its `--json` output) |
| `test_normality.py` | tests of the Python interface, with a brute force for small `n` |
