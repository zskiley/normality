"""Tests for the Python interface:  python -m unittest test_normality -v"""
import random
import unittest

import normality


def all_subspaces(n):
    """All linear subspaces of F_2^n, as bitmasks over the 2^n points, by dimension."""
    levels = [{1}]
    for _ in range(n):
        nxt = set()
        for u in levels[-1]:
            for v in range(1, 1 << n):
                if not (u >> v) & 1:
                    shifted = 0
                    for x in range(1 << n):
                        if (u >> x) & 1:
                            shifted |= 1 << (x ^ v)
                    nxt.add(u | shifted)
        levels.append(nxt)
    return levels


def brute_force(bits, n):
    """Largest k with f constant on a k-flat, by trying every flat."""
    ones = sum(1 << x for x in range(1 << n) if bits[x])
    levels = all_subspaces(n)
    for k in range(n, -1, -1):
        for u in levels[k]:
            for a in range(1 << n):
                flat = 0
                for x in range(1 << n):
                    if (u >> x) & 1:
                        flat |= 1 << (x ^ a)
                inside = flat & ones
                if inside == flat or inside == 0:
                    return k
    return -1


class Small(unittest.TestCase):
    def test_against_brute_force(self):
        rng = random.Random(1)
        for n in (2, 3, 4, 5):
            for _ in range(12):
                bits = [rng.randrange(2) for _ in range(1 << n)]
                r = normality.solve(bits)
                self.assertEqual(r.k, brute_force(bits, n), (n, bits))
                self.assertTrue(r.exact)
                self.assertTrue(normality.check(bits, r))

    def test_constant(self):
        r = normality.solve([0] * 32)
        self.assertEqual((r.k, r.value), (5, 0))
        self.assertEqual(sorted(r.points()), list(range(32)))

    def test_quadratic_bent(self):
        f = lambda x: (x & 1) & (x >> 1 & 1) ^ (x >> 2 & 1) & (x >> 3 & 1)   # x0 x1 + x2 x3
        r = normality.solve(f, n=4)
        self.assertEqual(r.k, 2)
        self.assertTrue(r.bent)
        self.assertTrue(r.normal)
        self.assertTrue(normality.check(f, r, n=4))


class Inputs(unittest.TestCase):
    def test_same_function_in_every_form(self):
        rng = random.Random(7)
        n = 8
        bits = [rng.randrange(2) for _ in range(1 << n)]
        as_int = sum(b << i for i, b in enumerate(bits))
        forms = [
            (bits, None),
            (tuple(bits), None),
            ("".join(map(str, bits)), None),
            (" ".join(map(str, bits)) + "\n", None),
            (bytes(bits), None),
            ("".join(map(str, bits)).encode(), None),
            (as_int, n),
            (lambda x: bits[x], n),
        ]
        try:
            import numpy as np
            forms.append((np.array(bits, dtype=np.uint8), None))
            forms.append((np.array(bits, dtype=bool), None))
        except ImportError:
            pass
        ks = {normality.solve(t, n).k for t, n in forms}
        self.assertEqual(len(ks), 1)
        self.assertEqual(normality.truth_table(lambda x: bits[x], n), bits)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            normality.solve([0, 1, 1])
        with self.assertRaises(ValueError):
            normality.solve("0120")
        with self.assertRaises(ValueError):
            normality.solve(lambda x: 0)
        with self.assertRaises(ValueError):
            normality.solve([0, 1, 1, 0], n=3)
        with self.assertRaises(ValueError):
            normality.family("nosuch", 8)


class Questions(unittest.TestCase):
    def test_target(self):
        rng = random.Random(3)
        for n in (9, 10):
            bits = [rng.randrange(2) for _ in range(1 << n)]
            k = normality.largest_k(bits)
            self.assertTrue(normality.is_k_normal(bits, k))
            self.assertFalse(normality.is_k_normal(bits, k + 1))
            r = normality.solve(bits, target=k + 1)
            self.assertFalse(r.exact)
            self.assertIs(r.target_normal, False)
            self.assertEqual(normality.is_normal(bits), k >= (n + 1) // 2)

    def test_family(self):
        r = normality.family("kasami", 14)
        self.assertEqual((r.n, r.k), (14, 6))
        self.assertTrue(r.bent)
        self.assertIs(r.normal, False)
        s = normality.family("kasami", 14, known_symmetry=True)
        self.assertEqual(s.k, 6)
        self.assertEqual(s.symmetry["order"], 21)
        self.assertTrue(normality.family("kasami", 10).normal)
        self.assertEqual(normality.family("kasami", 10, plus=1, hyp=1).k, 5)

    def test_maps(self):
        n = 9
        maj = normality.truth_table(lambda x: bin(x).count("1") >= 5, n)
        swaps = []
        for i in range(n - 1):       # the coordinate transpositions (i, i+1)
            cols = [1 << j for j in range(n)]
            cols[i], cols[i + 1] = cols[i + 1], cols[i]
            swaps.append(cols + [0])
        not_one = [1 << ((j + 1) % n) if j < 2 else 1 << j for j in range(n)] + [5]   # not an automorphism
        plain = normality.solve(maj)
        r = normality.solve(maj, maps=swaps + [not_one])
        self.assertEqual(r.k, plain.k)
        self.assertEqual(r.symmetry["generators"], n - 1)
        self.assertTrue(normality.check(maj, r))


def add_p(x, y, p, n):
    """x + y in F_p^n on the integers x_0 + x_1 p + ..."""
    return sum((x // p ** i + y // p ** i) % p * p ** i for i in range(n))


def brute_force_p(vals, p, n):
    """Largest k with f constant on a k-flat of F_p^n, by trying every flat."""
    size = p ** n
    levels = [{frozenset([0])}]
    for _ in range(n):
        nxt = set()
        for u in levels[-1]:
            for v in range(1, size):
                if v not in u:
                    w, tv = set(), 0
                    for _t in range(p):
                        w.update(add_p(x, tv, p, n) for x in u)
                        tv = add_p(tv, v, p, n)
                    nxt.add(frozenset(w))
        levels.append(nxt)
    for k in range(n, -1, -1):
        for u in levels[k]:
            for a in range(size):
                if len({vals[add_p(a, x, p, n)] for x in u}) == 1:
                    return k
    return -1


class Defaults(unittest.TestCase):
    def test_symmetry_is_used_by_default(self):
        off = normality.family("inverse", 13, known_symmetry=False, aut=False)
        on = normality.family("inverse", 13)
        self.assertEqual(on.k, off.k)
        self.assertEqual(off.symmetry["generators"], 0)
        self.assertGreater(on.symmetry["generators"], 0)
        self.assertGreater(on.symmetry["skipped"], 0)
        rng = random.Random(4)
        bits = [rng.randrange(2) for _ in range(1 << 10)]
        self.assertEqual(normality.solve(bits).k, normality.solve(bits, aut=False).k)


class OddCharacteristic(unittest.TestCase):
    def test_against_brute_force(self):
        rng = random.Random(5)
        for p, n, reps in ((3, 1, 3), (3, 2, 8), (3, 3, 8), (5, 2, 6), (7, 2, 3)):
            for it in range(reps):
                vals = [rng.randrange(p) if it % 2 else rng.randrange(2) for _ in range(p ** n)]
                r = normality.solve(vals, p=p)
                self.assertEqual(r.k, brute_force_p(vals, p, n), (p, n, vals))
                self.assertEqual((r.p, r.n, len(r.upper_bounds)), (p, n, p))
                self.assertTrue(r.exact)
                self.assertTrue(normality.check(vals, r))
                self.assertEqual(len(set(r.points())), p ** r.k)

    def test_quadratic_forms(self):
        hyp = lambda x: (x % 3) * (x // 3 % 3) + (x // 9 % 3) * (x // 27 % 3)      # x0 x1 + x2 x3
        ell = lambda x: (x % 3) ** 2 + (x // 3 % 3) ** 2 + (x // 9 % 3) * (x // 27 % 3)   # x0^2 + x1^2 + x2 x3
        r = normality.solve(hyp, n=4, p=3)
        self.assertEqual((r.k, r.value), (2, 0))
        self.assertTrue(r.bent)
        self.assertTrue(r.normal)
        self.assertTrue(normality.check(hyp, r, n=4))
        r = normality.solve(ell, n=4, p=3)
        self.assertEqual(r.k, 1)
        self.assertTrue(r.bent)
        self.assertIs(r.normal, False)
        self.assertEqual(r.coordinates(r.offset + 0), tuple(r.offset // 3 ** i % 3 for i in range(4)))

    def test_same_function_in_every_form(self):
        rng = random.Random(11)
        for p, n in ((3, 4), (11, 2)):
            vals = [rng.randrange(p) for _ in range(p ** n)]
            text = "".join(map(str, vals)) if p <= 7 else " ".join(map(str, vals))
            forms = [(vals, None), (tuple(vals), None), (text, None), (text.encode(), None), (lambda x: vals[x], n)]
            if p > 7:
                forms.append((",".join(map(str, vals)), None))
            try:
                import numpy as np
                forms.append((np.array(vals, dtype=np.uint8), None))
            except ImportError:
                pass
            ks = {normality.solve(t, m, p=p).k for t, m in forms}
            self.assertEqual(len(ks), 1)
            self.assertEqual(normality.truth_table(lambda x: vals[x], n, p), vals)

    def test_questions_and_families(self):
        rng = random.Random(2)
        vals = [rng.randrange(2) for _ in range(3 ** 5)]
        k = normality.largest_k(vals, p=3)
        self.assertTrue(normality.is_k_normal(vals, k, p=3))
        self.assertFalse(normality.is_k_normal(vals, k + 1, p=3))
        self.assertEqual(normality.is_normal(vals, p=3), k >= 3)
        r = normality.family("power", 10, 14, p=3)          # Tr(x^14) over GF(3^10): bent and normal
        self.assertEqual((r.p, r.n, r.k), (3, 10, 5))
        self.assertTrue(r.bent and r.normal)
        self.assertEqual(normality.family("quad", 8, p=3).k, 4)
        self.assertEqual(normality.family("quad", 8, 1, p=3).k, 3)
        self.assertEqual(normality.family("quad", 6, 1, p=5, scramble=3).k, 2)
        self.assertEqual(normality.family("mm", 6, p=5).k, 3)
        two = normality.family("power", 4, 2, 0, p=3, more=[(4, 1)])     # Tr(x^2 + g x^4), also a quadratic form
        self.assertEqual(two.n, 4)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            normality.solve([0, 1, 2, 0], p=3)
        with self.assertRaises(ValueError):
            normality.solve("0123", p=4)
        with self.assertRaises(ValueError):
            normality.solve("012012013", p=3)
        with self.assertRaises(ValueError):
            normality.solve([0, 1, 2] * 3, p=3, aut=True)
        with self.assertRaises(ValueError):
            normality.family("kasami", 8, p=3)

    def test_symmetry(self):
        plain = normality.family("power", 8, 14, 1, p=3, known_symmetry=False)
        self.assertFalse(plain.symmetry["looked_at"])
        for sym in (normality.family("power", 8, 14, 1, p=3), normality.family("power", 8, 14, 1, p=3, known_symmetry=True)):
            self.assertEqual(sym.k, plain.k)       # used by default
            self.assertEqual(sym.symmetry["generators"], 2)
            self.assertGreater(sym.symmetry["skipped"], 0)
        # a function of the cyclic classes of (x_0, ..., x_4) over F_3: the rotation is a symmetry
        n, p = 5, 3
        rot = lambda x: x // p + (x % p) * p ** (n - 1)
        def cls(x):
            best = x
            for _ in range(n):
                x = rot(x)
                best = min(best, x)
            return best
        vals = [(cls(x) * 7 + cls(x) // 5) % 2 for x in range(p ** n)]
        maps = [[p ** ((j - 1) % n) for j in range(n)] + [0],          # the rotation
                [p ** ((j + 1) % n) + (j == 0) for j in range(n)]]     # not a symmetry
        plain = normality.solve(vals, p=p)
        r = normality.solve(vals, p=p, maps=maps)
        self.assertEqual(r.k, plain.k)
        self.assertTrue(normality.check(vals, r))
        if r.symmetry["looked_at"]:
            self.assertEqual(r.symmetry["generators"], 1)


if __name__ == "__main__":
    unittest.main()
