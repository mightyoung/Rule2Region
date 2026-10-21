"""Independent exhaustive oracle for SYNTHETIC bipartite matching graphs.

No pixels, private returns, models, human labels or research measurements.
The oracle enumerates assignments directly; it shares no optimization code with
the production matcher. Removing any objective level must break these tests.
"""
import importlib
import random
import unittest
from fractions import Fraction


def exhaustive_matching(edges):
    """Choose exactly the approved three-level optimum by exhaustive search."""
    ground_truth_ids = sorted({g for g, _ in edges})
    best = None
    best_pairs = None

    def visit(index, used_predictions, pairs, total_iou):
        nonlocal best, best_pairs
        if index == len(ground_truth_ids):
            ordered_pairs = tuple(sorted(pairs))
            objective = (-len(ordered_pairs), -total_iou, ordered_pairs)
            if best is None or objective < best:
                best = objective
                best_pairs = ordered_pairs
            return
        gt_id = ground_truth_ids[index]
        visit(index + 1, used_predictions, pairs, total_iou)
        for pair, iou in edges.items():
            if pair[0] == gt_id and pair[1] not in used_predictions:
                visit(index + 1, used_predictions | {pair[1]},
                      pairs + [pair], total_iou + iou)

    visit(0, set(), [], Fraction(0))
    return list(best_pairs)


class SyntheticMatchingOracle(unittest.TestCase):
    def setUp(self):
        try:
            module = importlib.import_module('scoring_contract')
        except ModuleNotFoundError as error:
            self.fail('maximum_matching implementation missing: ' + str(error))
        self.assertTrue(callable(getattr(module, 'maximum_matching', None)),
                        'maximum_matching implementation missing')
        self.matcher = module.maximum_matching

    def assert_matches_oracle(self, edges):
        expected = exhaustive_matching(edges)
        actual = self.matcher(edges)
        self.assertEqual(actual, expected, repr(edges))
        self.assertEqual(actual, sorted(actual))
        self.assertEqual(len({g for g, _ in actual}), len(actual))
        self.assertEqual(len({p for _, p in actual}), len(actual))

    def test_synthetic_empty_graph(self):
        self.assertEqual(self.matcher({}), [])

    def test_synthetic_cardinality_precedes_total_iou_greedy_counterexample(self):
        edges = {('g1', 'p1'): Fraction(1),
                 ('g2', 'p2'): Fraction(1),
                 ('g1', 'p3'): Fraction(1, 2),
                 ('g2', 'p1'): Fraction(1, 2),
                 ('g3', 'p2'): Fraction(1, 2)}
        self.assertEqual(exhaustive_matching(edges),
                         [('g1', 'p3'), ('g2', 'p1'), ('g3', 'p2')])
        self.assert_matches_oracle(edges)

    def test_synthetic_total_iou_precedes_lexicographic_preference(self):
        edges = {('g1', 'p1'): Fraction(1, 2),
                 ('g1', 'p2'): Fraction(1),
                 ('g2', 'p1'): Fraction(1),
                 ('g2', 'p2'): Fraction(1, 2)}
        self.assertEqual(exhaustive_matching(edges), [('g1', 'p2'), ('g2', 'p1')])
        self.assert_matches_oracle(edges)

    def test_synthetic_exact_rational_difference_is_not_float_tie(self):
        # float(1/2 + 1/10**30) == float(1/2), but IoU objective differs.
        edges = {('g1', 'p1'): Fraction(1, 2),
                 ('g1', 'p2'): Fraction(1, 2) + Fraction(1, 10**30),
                 ('g2', 'p1'): Fraction(1, 2),
                 ('g2', 'p2'): Fraction(1, 2)}
        self.assertEqual(exhaustive_matching(edges), [('g1', 'p2'), ('g2', 'p1')])
        self.assert_matches_oracle(edges)

    def test_synthetic_full_ties_are_id_deterministic_under_reordering(self):
        edges = {(g, p): Fraction(3, 4)
                 for g in ['g-z', 'g-a', 'g-m']
                 for p in ['p-z', 'p-a', 'p-m', 'p-0']}
        expected = [('g-a', 'p-0'), ('g-m', 'p-a'), ('g-z', 'p-m')]
        self.assertEqual(exhaustive_matching(edges), expected)
        randomizer = random.Random(20261021)
        items = list(edges.items())
        for permutation in range(30):
            randomizer.shuffle(items)
            with self.subTest(permutation=permutation):
                self.assertEqual(self.matcher(dict(items)), expected)

    def test_synthetic_fixed_seed_small_graphs_match_exhaustive_oracle(self):
        randomizer = random.Random(2125)
        for case in range(300):
            gs = [f'g-{i}' for i in range(randomizer.randrange(5))]
            ps = [f'p-{i}' for i in range(randomizer.randrange(5))]
            edges = {(g, p): Fraction(randomizer.randrange(5, 11), 10)
                     for g in gs for p in ps if randomizer.randrange(3) != 0}
            items = list(edges.items())
            randomizer.shuffle(items)
            with self.subTest(synthetic_case=case):
                self.assert_matches_oracle(dict(items))

    def test_synthetic_disconnected_concave_roi_matches_rectangle_union_oracle(self):
        module = importlib.import_module('scoring_contract')
        # U shape: three disjoint-interior rectangles, so exact overlap area
        # is a sum. A horizontal clip above y=2 has two disconnected pieces.
        polygon = [[0,0],[6,0],[6,6],[4,6],[4,2],[2,2],[2,6],[0,6],[0,0]]
        cells = [(0,0,6,2), (0,2,2,6), (4,2,6,6)]
        randomizer = random.Random(20261022)
        boxes = [(0,3,6,5), (2,3,4,5), (2,2,4,6),
                 (7,1,8,2), (0,0,6,6)]
        for _ in range(200):
            x1, x2 = sorted(randomizer.sample(range(-2, 9), 2))
            y1, y2 = sorted(randomizer.sample(range(-2, 9), 2))
            boxes.append((x1,y1,x2,y2))
        normalized = module.roi_polygon(dict(version='synthetic-U',points=polygon), 10, 10)
        for coords in boxes:
            a,b,c,d = coords
            overlap = sum(max(0,min(c,z)-max(a,x)) * max(0,min(d,w)-max(b,y))
                          for x,y,z,w in cells)
            touches = any(min(c,z)>=max(a,x) and min(d,w)>=max(b,y)
                          for x,y,z,w in cells)
            expected = ('in' if overlap==(c-a)*(d-b) else 'boundary') if overlap else (
                'touch' if touches else 'out')
            rational = list(map(Fraction,coords))
            with self.subTest(synthetic_box=coords):
                self.assertEqual(module.relation(rational,normalized), expected)
                self.assertEqual(module.relation(rational,list(reversed(normalized))), expected)


if __name__ == '__main__':
    unittest.main()
