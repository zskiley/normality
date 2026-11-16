# normality

Finds the largest `k` such that a Boolean function `f : F_2^n -> F_2` is
**k-normal**, i.e. constant on some `k`-dimensional affine subspace (flat).
The answer is exact and comes with a flat that attains it. Practical for `n`
up to about 20.

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
options `aut=True` and `maps=[...]` described below.

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
That build has everything except `--aut`.

## Symmetry

If automorphisms of `f` are known, the search looks at one flat per orbit,
which can be much faster. They can come from three places:

- `--aut` / `aut=True`: computed by the
  [equivalence library](https://github.com/zskiley/Affine-and-linear-equivalence-algorithms)
  (the submodule in `third_party/aleq`) while the search runs. It does not
  finish for bent functions in 12 or more variables.
- `--aut-known` / `known_symmetry=True`: the symmetries known from the
  construction of a built-in family.
- `--aut-file FILE` / `maps=[...]`: maps you supply, each as the images of
  `e_0 .. e_(n-1)` followed by the image of 0.

Every map is checked against the truth table before it is used.

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
are in the comments at the top of `solver.hpp`, `search.hpp` and `group.hpp`.
