# normality

Finds the largest `k` such that a Boolean function `f : F_2^n -> F_2` is
**k-normal**, i.e. constant on some `k`-dimensional affine subspace (flat).
The answer is exact and comes with a flat that attains it. Practical for `n`
up to about 20.

Functions `f : F_p^n -> F_p` for an odd prime `p` (up to 61) are handled too,
see [Odd characteristic](#odd-characteristic).

## Install

```bash
git clone --recursive https://github.com/zskiley/normality
```

You need `g++`, and Python 3.8 or newer for the Python interface. The solver
is compiled automatically the first time you use it from Python.

## Python

```python
import normality

table = "0110100110010110"        # character i is f(i); bit j of i is the variable x_j
r = normality.solve(table)
r.k                               # the largest k
r.value, r.offset, r.basis        # f == value on the flat offset + span(basis)
list(r.points())                  # the points of that flat

normality.solve([0, 1, 1, 0, 1, 0, 0, 1])                 # list, tuple or numpy array
normality.solve(lambda x: bin(x).count("1") >= 5, n=9)    # a Python function of the point x
normality.is_k_normal(table, 3)   # decide one k only (faster than the maximum)
normality.is_normal(table)        # k = ceil(n/2)
normality.family("kasami", 14)    # a built-in family; normality.FAMILIES lists them
```

`solve` also takes `target=k`, `threads=`, `verbose=True`, and the symmetry
options `aut=` and `maps=[...]` described below.

## Command line

```bash
python normality.py build
```

```bash
./normality 0110100110010110          # truth table, character i = f(i)
./normality -f table.txt              # table from a file
./normality --kasami 14               # a built-in family
./normality --kasami 14 --target 7    # only decide "is f 7-normal?"
./normality --help                    # all options
```

To compile by hand without Python:
`g++ -O3 -march=native -std=c++17 -pthread -o normality main.cpp`
(with MinGW-w64 on Windows add `-static -Wa,-muse-unaligned-vector-move`).
That build has everything except the equivalence library.

## Symmetry

With automorphisms of `f` the search looks at one flat per orbit, which can
be much faster. Two sources are used by default:

- the automorphism group computed by the
  [equivalence library](https://github.com/zskiley/Affine-and-linear-equivalence-algorithms)
  (the submodule in `third_party/aleq`), in the background, once a search has
  taken more than a moment. It does not finish for bent functions in 12 or
  more variables. Off with `--no-aut` / `aut=False`.
- the symmetries known from the construction of a built-in family. Off with
  `--no-aut-known` / `known_symmetry=False`.

You can also supply maps yourself with `--aut-file FILE` / `maps=[...]`, each
as the images of `e_0 .. e_(n-1)` followed by the image of 0.

Every map is checked against the truth table before it is used.

## Odd characteristic

With `p=` (Python) or `--p` (command line) the function goes from `F_p^n` to
`F_p`. The table then has `p^n` values `0 .. p-1`; value number
`x_0 + x_1 p + ... + x_(n-1) p^(n-1)` is `f(x_0, ..., x_(n-1))`, and points
and basis vectors are reported as such integers.

```python
r = normality.solve("012120201", p=3)             # f(x_0, x_1) = x_0 + x_1 over F_3
r.k, r.value, r.offset, r.basis
r.coordinates(r.basis[0])                         # (x_0, ..., x_(n-1)) of a vector
normality.solve(lambda x: (x % 5) * (x // 5 % 5) % 5, n=2, p=5)   # x_0 x_1 over F_5
normality.is_k_normal(table, 3, p=3)
normality.family("power", 10, 14, p=3)            # Tr(x^14) over GF(3^10); see normality.PFAMILIES
```

```bash
./normality --p 3 012120201
```

```bash
./normality --p 3 --power 10 14
```

For `p <= 7` a table given as text has one digit per value; for larger `p`
the values are numbers separated by spaces or commas. Tables of up to a few
million values are practical (for `p = 3` that is `n` up to 12 to 14).

Symmetry works here as well, but the library is binary only: by default it
is used for the `power` family (the maps `x -> a x^(p^i)` of the field that
are symmetries), and for a table only if you supply maps with `--aut-file` /
`maps=[...]` as above, with points as integers. A map may also permute the
values of `f`, i.e. `f(A x) = s(f(x))`; then only one of the constants that
`s` links is searched.

## Tests

```bash
./normality --selftest
```

```bash
python -m unittest test_normality
```

## How it works

An upper bound from the Walsh spectrum, a greedy lower bound, then an
exhaustive search over direction spaces using bitset quotients. The details
are in the comments at the top of `solver.hpp`, `search.hpp` and `group.hpp`,
and of `pary.hpp` for odd `p`.
