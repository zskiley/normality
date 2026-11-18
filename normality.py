"""Python interface to the normality solver.

A Boolean function f on n variables is k-normal if it is constant on some
k-dimensional affine subspace (flat) of F_2^n.  This module finds the largest
such k, exactly, together with a flat that attains it.

    import normality

    r = normality.solve("0110100110010110")      # truth table: character i is f(i)
    print(r.k, r.value, r.offset, r.basis)       # f == r.value on r.offset + span(r.basis)

    r = normality.solve(lambda x: bin(x).count("1") >= 5, n=9)   # a Python function
    normality.is_k_normal(table, 7)              # only decide one k (much cheaper)
    normality.family("kasami", 14)               # a built-in family

Points are integers: bit j of x is the variable x_j, and the truth table lists
f(0), f(1), ..., f(2^n - 1).

Functions F_p^n -> F_p for an odd prime p (up to 61) work the same way with
p=...: the table has p^n values 0 .. p-1, and the point x_0 + x_1 p + ... has
the coordinates x_0, x_1, ...

    r = normality.solve("012120201", p=3)        # f(x_0, x_1) = x_0 + x_1 over F_3
    normality.solve(lambda x: (x % 3) * (x // 3 % 3) % 3, n=2, p=3)
    normality.family("power", 10, 14, p=3)       # Tr(x^14) over GF(3^10)

The work is done by the C++ program in this folder.  It is compiled on first
use (g++ is needed), or explicitly with normality.build().  The equivalence
library in third_party/aleq (a git submodule) is built in if it is there.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Tuple, Union

__all__ = [
    "solve", "largest_k", "is_k_normal", "is_normal", "family", "truth_table", "check",
    "build", "binary", "Result", "NormalityError", "FAMILIES", "PFAMILIES",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXE = "normality.exe" if os.name == "nt" else "normality"

# Built-in families and the numbers they take (see `normality --help`).
FAMILIES = {
    "kasami": "N [K]     Kasami bent function Tr(a x^(4^K-2^K+1)), a of order 3 (K = 3 by default)",
    "dillon": "N         Dillon bent function Tr(a x^(2^(N/2)-1))",
    "mm": "N [SEED]  Maiorana-McFarland bent function, scrambled",
    "d0": "N [SEED]  Maiorana-McFarland class D0 bent function, scrambled",
    "ps": "N [SEED]  partial-spread bent function, scrambled",
    "power": "N D [E]   Tr(g^E x^D) over GF(2^N)",
    "inverse": "N         Tr(x^-1)",
    "aes": "MASK      component <MASK, S(x)> of the AES S-box",
    "cf": "N         Carlet-Feng function",
    "maj": "N         majority",
    "threshold": "N T       wt(x) >= T",
    "hwb": "N         hidden weighted bit function",
    "random": "N [SEED]  uniformly random function",
}

# The families available with p=... (an odd prime).
PFAMILIES = {
    "power": "N D [E]   Tr(g^E x^D) over GF(p^N), g primitive; more=[(D, E), ...] adds further terms",
    "mm": "N [SEED]  Maiorana-McFarland bent function",
    "quad": "N [T]     quadratic form x0 x1 + x2 x3 + ... (T = 1: the elliptic one)",
    "random": "N [SEED]  uniformly random function",
}

Table = Union[str, int, bytes, Iterable[Any], Callable[[int], Any]]


class NormalityError(RuntimeError):
    """The solver could not be built or run."""


@dataclass(frozen=True)
class Result:
    """What the solver found.

    f == value on the flat  offset + span(basis), which has dimension k.
    If `exact` is true, k is the largest dimension of any flat on which f is
    constant.  It is false only when a `target` was given: then the solver
    stopped as soon as the question "is f target-normal?" was settled, and k
    is just the dimension of the best flat it had found.
    """

    n: int
    k: int
    value: int
    offset: int
    basis: Tuple[int, ...]
    exact: bool
    upper_bounds: Tuple[int, ...]   # spectral upper bounds for f = 0, f = 1, ...
    bent: bool
    max_walsh: float
    searches: int                   # exhaustive searches that were needed
    seconds: float
    symmetry: dict                  # what the symmetry options contributed
    target: Optional[int] = None
    target_normal: Optional[bool] = None
    p: int = 2                      # the function goes from F_p^n to F_p
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def normal(self) -> Optional[bool]:
        """Is f constant on a flat of dimension ceil(n/2)?  None if that was not decided."""
        half = (self.n + 1) // 2
        if self.exact or self.k >= half:
            return self.k >= half
        if self.target is not None and self.target <= half:
            return False if self.target_normal is False else None
        return None

    def points(self) -> Iterator[int]:
        """The p^k points of the flat."""
        x = self.offset
        yield x
        if self.p == 2:
            for g in range(1, 1 << self.k):
                x ^= self.basis[(g & -g).bit_length() - 1]   # Gray code: one basis vector per step
                yield x
            return
        count = [0] * self.k
        while True:
            for i in range(self.k):                # odometer: p times a vector is zero
                x = _add(x, self.basis[i], self.p, self.n)
                count[i] += 1
                if count[i] < self.p:
                    break
                count[i] = 0
            else:
                return
            yield x

    def coordinates(self, x: int) -> Tuple[int, ...]:
        """The coordinates (x_0, ..., x_(n-1)) of a point or vector given as an integer."""
        return tuple(x // self.p ** i % self.p for i in range(self.n))


def _add(x: int, y: int, p: int, n: int) -> int:
    """x + y in F_p^n, on integers x_0 + x_1 p + ..."""
    r, w = 0, 1
    for _ in range(n):
        r += (x % p + y % p) % p * w
        x //= p
        y //= p
        w *= p
    return r


# ---------------------------------------------------------------- the binary

def binary() -> Optional[str]:
    """Path of the solver executable, or None if there is none yet."""
    env = os.environ.get("NORMALITY_BIN")
    if env and os.path.isfile(env):
        return env
    here = os.path.join(_HERE, _EXE)
    if os.path.isfile(here):
        return here
    return shutil.which("normality")


def build(with_library: Optional[bool] = None, compiler: str = "g++", extra_flags: Sequence[str] = (),
          quiet: bool = False) -> str:
    """Compile the solver next to this file and return its path.

    with_library: also build in the external equivalence library (needed for
    aut=True).  By default that is done if the submodule third_party/aleq has
    been fetched; should that build fail (the library needs C++20), the solver
    is built without it.
    """
    if not os.path.isfile(os.path.join(_HERE, "main.cpp")):
        raise NormalityError("the C++ sources are not next to normality.py; build the solver yourself and "
                             "set the environment variable NORMALITY_BIN to it")
    if shutil.which(compiler) is None:
        raise NormalityError(f"compiler '{compiler}' not found; install g++ or pass compiler=...")
    lib = os.path.join("third_party", "aleq", "src")
    have_lib = os.path.isfile(os.path.join(_HERE, lib, "equivalence_api.cpp"))
    auto = with_library is None
    if auto:
        with_library = have_lib
    if with_library and not have_lib:
        raise NormalityError("the equivalence library is not in third_party/aleq; "
                             "fetch it with:  git submodule update --init")
    cmd = [compiler, "-O3", "-march=native", "-pthread"]
    if os.name == "nt":   # MinGW-w64: see README.md for why these two are needed
        cmd += ["-static", "-Wa,-muse-unaligned-vector-move"]
    cmd += list(extra_flags)
    if with_library:
        cmd += ["-std=c++20", "-DNORMALITY_ALEQ", "-I", lib, "-o", _EXE, "main.cpp", "autgroup.cpp",
                os.path.join(lib, "dfs.cpp"), os.path.join(lib, "equivalence_api.cpp")]
    else:
        cmd += ["-std=c++17", "-o", _EXE, "main.cpp"]
    if not quiet:
        print("normality: compiling the solver (" + ("with" if with_library else "without")
              + " the equivalence library) ...", file=sys.stderr)
    done = subprocess.run(cmd, cwd=_HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if done.returncode != 0:
        if auto and with_library:   # most likely a compiler without C++20
            if not quiet:
                print("normality: that did not compile; trying without the library ...", file=sys.stderr)
            return build(False, compiler, extra_flags, quiet)
        raise NormalityError("compilation failed:\n" + " ".join(cmd) + "\n" + done.stdout)
    return os.path.join(_HERE, _EXE)


def _binary() -> str:
    exe = binary()
    return exe if exe else build()


# ------------------------------------------------------------- truth tables

def truth_table(f: Callable[[int], Any], n: int, p: int = 2) -> list:
    """[f(0), f(1), ..., f(p^n - 1)].  For p = 2 the values are 0/1 and bit j of
    the argument is x_j; otherwise they are taken modulo p and the argument is
    x_0 + x_1 p + ..."""
    if p == 2:
        return [1 if f(x) else 0 for x in range(1 << n)]
    return [int(f(x)) % p for x in range(p ** n)]


def _prime(p: int) -> int:
    p = int(p)
    if p < 2 or p > 61 or any(p % d == 0 for d in range(2, p)):
        raise ValueError("p must be a prime between 2 and 61")
    return p


def _values(table: Table, n: Optional[int], p: int) -> list:
    """Any accepted form of the table of a function F_p^n -> F_p (p odd) as a list of p^n values."""
    if callable(table):
        if n is None:
            raise ValueError("give n=... when the function is a Python callable")
        vals = [int(table(x)) % p for x in range(p ** n)]
    elif isinstance(table, int) and not isinstance(table, bool):
        raise ValueError("an integer truth table is only possible for p = 2")
    else:
        if isinstance(table, (bytes, bytearray)) and all(c in b"0123456789 \t\r\n," for c in table):
            table = table.decode("ascii")
        if isinstance(table, str):
            if p <= 7:      # one digit per value
                vals = [int(c) if c.isdigit() else -1 for c in table if c not in " \t\r\n,"]
            else:           # numbers separated by white space or commas
                vals = [int(t) if t.isdigit() else -1 for t in table.replace(",", " ").split()]
        else:
            vals = [int(v) for v in table]
        if any(v < 0 or v >= p for v in vals):
            raise ValueError(f"the values of the table must be between 0 and {p - 1}")
    size, nvar = len(vals), 0
    while p ** nvar < size:
        nvar += 1
    if size < p or p ** nvar != size:
        raise ValueError(f"the table needs {p}^n values with n >= 1, got {size}")
    if n is not None and nvar != n:
        raise ValueError(f"the table has {size} values, which is not {p}^{n}")
    return vals


def _nvar(table: Table, n: Optional[int], p: int) -> int:
    if p == 2:
        return len(_bits(table, n)).bit_length() - 1
    size, nvar = len(_values(table, n, p)), 0
    while p ** nvar < size:
        nvar += 1
    return nvar


def _bits(table: Table, n: Optional[int]) -> str:
    """Any accepted form of a truth table as a string of 2^n characters 0/1."""
    if callable(table):
        if n is None:
            raise ValueError("give n=... when the function is a Python callable")
        s = "".join("1" if table(x) else "0" for x in range(1 << n))
    elif isinstance(table, int) and not isinstance(table, bool):
        if n is None:
            raise ValueError("give n=... when the truth table is an integer (bit i = f(i))")
        if table < 0 or table >> (1 << n):
            raise ValueError("the integer does not fit a truth table on n variables")
        s = bin(table)[2:].zfill(1 << n)[::-1]
    else:
        if isinstance(table, (bytes, bytearray)) and all(c in b"01 \t\r\n," for c in table):
            table = table.decode("ascii")
        if isinstance(table, str):
            s = "".join(c for c in table if c not in " \t\r\n,")
            if any(c not in "01" for c in s):
                raise ValueError("a truth table string may contain only 0 and 1")
        else:
            s = "".join("1" if v else "0" for v in table)
    size = len(s)
    if size < 2 or size & (size - 1):
        raise ValueError(f"a truth table needs 2^n values with n >= 1, got {size}")
    if n is not None and size != 1 << n:
        raise ValueError(f"the truth table has {size} values, which is not 2^{n}")
    return s


# -------------------------------------------------------------------- solving

def _run(args: Sequence[str], verbose: bool) -> Result:
    cmd = [_binary(), "--json"] + list(args)
    done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None if verbose else subprocess.PIPE, text=True)
    try:
        d = json.loads(done.stdout)
    except ValueError:
        d = None
    if done.returncode != 0 or not isinstance(d, dict) or d.get("k", -1) < 0:
        raise NormalityError("the solver failed (exit code %d): %s"
                             % (done.returncode, (done.stderr or done.stdout or "").strip()))
    return Result(
        n=d["n"], k=d["k"], value=d["value"], offset=d["offset"], basis=tuple(d["basis"]), exact=d["exact"],
        upper_bounds=tuple(d["upper_bounds"]), bent=d["bent"], max_walsh=d["max_walsh"], searches=d["searches"],
        seconds=d["seconds"], symmetry=d["symmetry"], target=d.get("target"), target_normal=d.get("target_normal"),
        p=d.get("p", 2), raw=d)


def _aut_warning(result: Result) -> Result:
    if "could not be started" in result.symmetry.get("note", ""):
        warnings.warn("aut=True had no effect: this solver was built without the equivalence library "
                      "(run 'git submodule update --init', then normality.build(); see README.md)", stacklevel=3)
    return result


def _options(target, threads, aut, aut_seconds, extra) -> list:
    args = []
    if target is not None:
        args += ["--target", str(int(target))]
    if threads is not None:
        args += ["-t", str(int(threads))]
    if aut:
        args += ["--aut", str(float(aut_seconds))]
    return args + [str(a) for a in extra]


def solve(table: Table, n: Optional[int] = None, *, p: int = 2, target: Optional[int] = None,
          threads: Optional[int] = None, aut: bool = False, aut_seconds: float = 60.0,
          maps: Optional[Iterable[Sequence[int]]] = None, verbose: bool = False,
          extra_args: Sequence[str] = ()) -> Result:
    """Largest k such that f is constant on a k-dimensional flat, with such a flat.

    table        the truth table: a string of 0/1, any sequence or numpy array
                 of 0/1, an integer whose bit i is f(i) (give n), or a Python
                 function of one integer (give n).
    p            an odd prime: f goes from F_p^n to F_p.  The table then has
                 p^n values 0 .. p-1 (a string of digits for p <= 7, of numbers
                 separated by spaces or commas for larger p, or a sequence, or
                 a Python function with n), and the point x_0 + x_1 p + ... has
                 the coordinates x_0, x_1, ...  The symmetry options aut and
                 maps are for p = 2 only.
    target       only decide whether f is target-normal; see Result.exact.
    threads      worker threads (default: all cores).
    aut          let the external equivalence library compute the automorphism
                 group while the search runs (for at most aut_seconds), and use
                 it to search one flat per orbit.  Needs a build with the
                 library.  It does not finish on bent functions in 12 or more
                 variables; see README.md.
    maps         automorphisms of f that you know: each one a sequence of n+1
                 integers, the images of the unit vectors e_0 .. e_(n-1)
                 followed by the image of 0.  Every map is verified by the
                 solver and ignored if it is not an automorphism.
    verbose      let the solver print its progress to stderr.
    """
    p = _prime(p)
    if p != 2:
        if aut or maps is not None:
            raise ValueError("the symmetry options are only available for p = 2")
        vals = _values(table, n, p)
        fd, path = tempfile.mkstemp(prefix="normality_", suffix=".tt")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(("" if p <= 7 else " ").join(str(v) for v in vals))
            args = ["--p", str(p), "-f", path] + (["-v"] if verbose else [])
            return _run(args + _options(target, threads, False, 0, extra_args), verbose)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    s = _bits(table, n)
    tmp = []
    try:
        fd, path = tempfile.mkstemp(prefix="normality_", suffix=".tt")
        tmp.append(path)
        with os.fdopen(fd, "w") as fh:
            fh.write(s)
        args = ["-f", path] + (["-v"] if verbose else []) + _options(target, threads, aut, aut_seconds, extra_args)
        if maps is not None:
            nvar = len(s).bit_length() - 1
            fd, mpath = tempfile.mkstemp(prefix="normality_", suffix=".maps")
            tmp.append(mpath)
            with os.fdopen(fd, "w") as fh:
                for mp in maps:
                    mp = [int(v) for v in mp]
                    if len(mp) not in (nvar, nvar + 1):
                        raise ValueError(f"a map needs {nvar} or {nvar + 1} integers, got {len(mp)}")
                    fh.write(" ".join(str(v) for v in mp) + "\n")
            args += ["--aut-file", mpath]
        return _aut_warning(_run(args, verbose))
    finally:
        for path in tmp:
            try:
                os.unlink(path)
            except OSError:
                pass


def largest_k(table: Table, n: Optional[int] = None, **options) -> int:
    """The largest k such that f is k-normal."""
    return solve(table, n, **options).k


def is_k_normal(table: Table, k: int, n: Optional[int] = None, **options) -> bool:
    """Is f constant on some k-dimensional flat?  Cheaper than computing the maximum."""
    return solve(table, n, target=k, **options).k >= k


def is_normal(table: Table, n: Optional[int] = None, **options) -> bool:
    """Is f normal, i.e. constant on a flat of dimension ceil(n/2)?"""
    nvar = _nvar(table, n, _prime(options.get("p", 2)))
    return is_k_normal(table, (nvar + 1) // 2, n, **options)


def family(name: str, *params: int, p: int = 2, more: Iterable[Sequence[int]] = (), dual: bool = False,
           plus: int = 0, hyp: int = 0, scramble: int = 0, known_symmetry: bool = False,
           target: Optional[int] = None, threads: Optional[int] = None, aut: bool = False,
           aut_seconds: float = 60.0, verbose: bool = False, extra_args: Sequence[str] = ()) -> Result:
    """Solve one of the built-in families (see FAMILIES), e.g. family("kasami", 14).

    The modifiers are applied in this order: dual (the dual of a bent
    function), plus=u (add the linear function <u, x>), hyp=r (add r hyperbolic
    pairs y1 y2 + ..., 2r more variables), scramble=seed (a random affine
    change of variables).  known_symmetry=True uses the symmetries known from
    the construction.

    With an odd prime p the families are those in PFAMILIES and the only
    modifier is scramble; family("power", N, D, E, p=p, more=[(D2, E2)]) is
    Tr(g^E x^D + g^E2 x^D2) over GF(p^N).
    """
    p = _prime(p)
    if p != 2:
        if name not in PFAMILIES:
            raise ValueError(f"unknown family '{name}' for p = {p}; one of: " + ", ".join(sorted(PFAMILIES)))
        if dual or plus or hyp or known_symmetry or aut:
            raise ValueError("for odd p the only modifier is scramble")
        args = ["--p", str(p), "--" + name] + [str(int(v)) for v in params]
        for term in more:
            args += ["--power", str(int(params[0]))] + [str(int(v)) for v in term]
        if scramble:
            args += ["--scramble", str(int(scramble))]
        if verbose:
            args.append("-v")
        return _run(args + _options(target, threads, False, 0, extra_args), verbose)
    if name not in FAMILIES:
        raise ValueError(f"unknown family '{name}'; one of: " + ", ".join(sorted(FAMILIES)))
    args = ["--" + name] + [str(int(v)) for v in params]
    if dual:
        args.append("--dual")
    if plus:
        args += ["--plus", str(int(plus))]
    if hyp:
        args += ["--hyp", str(int(hyp))]
    if scramble:
        args += ["--scramble", str(int(scramble))]
    if known_symmetry:
        args.append("--aut-known")
    if verbose:
        args.append("-v")
    return _aut_warning(_run(args + _options(target, threads, aut, aut_seconds, extra_args), verbose))


def check(table: Table, result: Result, n: Optional[int] = None) -> bool:
    """Verify in Python that f == result.value on the flat and that the flat has dimension result.k."""
    if result.p != 2:
        p = result.p
        vals = _values(table, n, p)
        if len(vals) != p ** result.n or len(result.basis) != result.k:
            return False
        rows = []                  # independence: elimination modulo p
        for v in result.basis:
            row = list(result.coordinates(v))
            for r in rows:
                lead = max(i for i, c in enumerate(r) if c)
                cf = row[lead] * pow(r[lead], -1, p) % p
                row = [(a - cf * b) % p for a, b in zip(row, r)]
            if not any(row):
                return False
            rows.append(row)
        return all(vals[x] == result.value for x in result.points())
    s = _bits(table, n)
    if len(s) != 1 << result.n or len(result.basis) != result.k:
        return False
    rows = []                      # the basis must be linearly independent
    for v in result.basis:
        for r in rows:
            v = min(v, v ^ r)
        if v == 0:
            return False
        rows.append(v)
    want = str(result.value)
    return all(s[x] == want for x in result.points())


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print(build())
    elif len(sys.argv) > 1:
        sys.exit(subprocess.run([_binary()] + sys.argv[1:]).returncode)
    else:
        print(__doc__)
