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


if __name__ == "__main__":
    unittest.main()
