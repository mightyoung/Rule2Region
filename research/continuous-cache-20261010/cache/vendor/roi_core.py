"""Pure-Python frozen Stage-A ROI routing; no detector or external dependencies.

Public geometry inputs use continuous original-image edge coordinates in
[0, width] x [0, height]. Convert normalized coordinates explicitly with
``normalized_to_pixels``; there is no coordinate-system guessing or clipping.

Predicates, ear triangulation and triangle/rectangle clipping use exact rational
arithmetic on the supplied binary floating-point coordinates. The geometric
policy tolerance is therefore zero: a real gap is not expanded into a touch,
and a representable positive sliver is not treated as empty. Input conversion
and returned areas are floating point; rational area underflow/overflow routes
to review_unknown. Geometry validity does not verify hashes, image identity,
ROI semantics or scene/rule versions: the caller must verify those separately.

boundary_touch means the closed box intersects the ROI boundary, including an
ROI enclosed by the box. boundary_crossing means the box has positive area
both inside and outside the ROI. A zero-area touch is retained for review.
"""
from fractions import Fraction
from math import isfinite
from numbers import Real
from collections.abc import Sequence

IMPLEMENTATION_VERSION = 'r2r-roi-exact-rational-v1'
GEOMETRY_TOLERANCE_PX = 0.0
TARGET_METRIC_NAMES = ('in_region_recall', 'in_region_recall_loss',
                       'boundary_retention', 'outside_suppression')


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError('non_numeric_coordinate')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError('non_finite_coordinate') from exc
    if not isfinite(result):
        raise ValueError('non_finite_coordinate')
    return result


def _dimensions(width, height):
    width, height = _number(width), _number(height)
    if width <= 0 or height <= 0:
        raise ValueError('non_positive_image_dimensions')
    return width, height


def _points(points):
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ValueError('invalid_polygon_schema')
    result = []
    for point in points:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            raise ValueError('invalid_point_schema')
        result.append((_number(point[0]), _number(point[1])))
    if len(result) > 1 and result[-1] == result[0]:
        result.pop()
    return result


def _rational(points):
    return [(Fraction(x), Fraction(y)) for x, y in points]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _signed_area2(points):
    if len(points) < 3:
        return Fraction(0)
    # Triangular fan around a local origin avoids unnecessary huge products.
    origin = points[0]
    return sum((_cross(origin, points[i], points[i + 1])
                for i in range(1, len(points) - 1)), Fraction(0))


def _float_area(value):
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError('area_not_representable') from exc
    if not isfinite(result) or (value > 0 and result == 0):
        raise ValueError('area_not_representable')
    return result


def area(poly):
    """Return unsigned shoelace area; this alone does not validate simplicity."""
    return _float_area(abs(_signed_area2(_rational(_points(poly)))) / 2)


def normalized_to_pixels(points, width, height):
    """Convert explicitly normalized points by x*W/y*H, never W-1/H-1.

    Reject out-of-range/non-finite data instead of silently clamping it.
    This conversion does not certify polygon topology; use polygon_validation.
    """
    width, height = _dimensions(width, height)
    points = _points(points)
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in points):
        raise ValueError('normalized_coordinate_out_of_bounds')
    return [(x * width, y * height) for x, y in points]


def _on_segment(point, a, b):
    return (_cross(a, b, point) == 0
            and min(a[0], b[0]) <= point[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= point[1] <= max(a[1], b[1]))


def _segments_intersect(a, b, c, d):
    ab_c, ab_d = _cross(a, b, c), _cross(a, b, d)
    cd_a, cd_b = _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > 0 > ab_d or ab_d > 0 > ab_c)
            and (cd_a > 0 > cd_b or cd_b > 0 > cd_a)):
        return True
    return ((ab_c == 0 and _on_segment(c, a, b))
            or (ab_d == 0 and _on_segment(d, a, b))
            or (cd_a == 0 and _on_segment(a, c, d))
            or (cd_b == 0 and _on_segment(b, c, d)))


def polygon_validation(points, width, height):
    """Validate a simple nonzero-area ring; an explicit closing point is allowed.

    Returns valid/reason/area_px2/points_px. Self-touching, overlapping edges,
    duplicate vertices, malformed coordinates and out-of-image points fail.
    Collinear vertices in a forward edge are accepted.
    """
    invalid = {'valid': False, 'reason': None, 'area_px2': None, 'points_px': None}
    try:
        width, height = _dimensions(width, height)
        parsed = _points(points)
        if len(parsed) < 3:
            raise ValueError('fewer_than_three_vertices')
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in parsed):
            raise ValueError('polygon_coordinate_out_of_bounds')
        if len(set(parsed)) != len(parsed):
            raise ValueError('duplicate_vertex')
        p = _rational(parsed)
        size = len(p)
        for i in range(size):
            prev, current, following = p[i - 1], p[i], p[(i + 1) % size]
            if _cross(prev, current, following) == 0 and not _on_segment(current, prev, following):
                raise ValueError('overlapping_adjacent_edges')
            for j in range(i + 1, size):
                if j == i + 1 or (i == 0 and j == size - 1):
                    continue  # Adjacent edges share their one allowed endpoint.
                if _segments_intersect(p[i], p[(i + 1) % size], p[j], p[(j + 1) % size]):
                    raise ValueError('self_intersection')
        exact_area = abs(_signed_area2(p)) / 2
        if exact_area == 0:
            raise ValueError('zero_polygon_area')
        numeric_area = _float_area(exact_area)
    except (ValueError, TypeError) as exc:
        invalid['reason'] = str(exc)
        return invalid
    return {'valid': True, 'reason': None, 'area_px2': numeric_area,
            'points_px': parsed}


def _triangulate(polygon):
    """Ear-clip an already validated simple ring using exact predicates."""
    p = list(polygon)
    # Remove redundant collinear vertices before testing ear containment.
    changed = True
    while changed and len(p) > 3:
        changed = False
        for i in range(len(p)):
            if _cross(p[i - 1], p[i], p[(i + 1) % len(p)]) == 0:
                del p[i]
                changed = True
                break
    if _signed_area2(p) < 0:
        p.reverse()
    triangles = []
    while len(p) > 3:
        for i, b in enumerate(p):
            previous, following = (i - 1) % len(p), (i + 1) % len(p)
            a, c = p[previous], p[following]
            if _cross(a, b, c) <= 0:
                continue
            if any(_cross(a, b, q) >= 0 and _cross(b, c, q) >= 0
                   and _cross(c, a, q) >= 0
                   for j, q in enumerate(p) if j not in (previous, i, following)):
                continue
            triangles.append((a, b, c))
            del p[i]
            break
        else:
            raise ValueError('triangulation_failed')
    triangles.append(tuple(p))
    return triangles


def _clip_triangle(triangle, box):
    """Sutherland-Hodgman clip one convex triangle to a closed rectangle."""
    points = list(triangle)
    for axis, bound, lower_bound in [(0, box[0], True), (0, box[2], False),
                                      (1, box[1], True), (1, box[3], False)]:
        if not points:
            break
        clipped = []
        a = points[-1]
        a_inside = a[axis] >= bound if lower_bound else a[axis] <= bound
        for b in points:
            b_inside = b[axis] >= bound if lower_bound else b[axis] <= bound
            if a_inside != b_inside:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                clipped.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if b_inside:
                clipped.append(b)
            a, a_inside = b, b_inside
        points = clipped
    return abs(_signed_area2(points)) / 2


def _boundary_contact(polygon, box):
    x1, y1, x2, y2 = box
    if any(x1 <= x <= x2 and y1 <= y <= y2 for x, y in polygon):
        return True
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return any(_segments_intersect(polygon[i - 1], polygon[i], corners[j - 1], corners[j])
               for i in range(len(polygon)) for j in range(4))


def _unknown(reason):
    return {'intersection_area_px2': None, 'boundary_touch': False,
            'boundary_crossing': False, 'decision': 'review_unknown',
            'reason': reason}


def route_box(box, poly, width, height):
    """Route a complete original-coordinate xyxy prediction against one ROI.

    Unknown flags are False because the relationship was not established;
    decision/reason must be checked before interpreting them as geometry.
    This function never alters/clips a prediction or infers behavioral labels.
    """
    try:
        width, height = _dimensions(width, height)
    except ValueError as exc:
        return _unknown('invalid_image_dimensions:' + str(exc))
    validation = polygon_validation(poly, width, height)
    if not validation['valid']:
        return _unknown('invalid_polygon:' + validation['reason'])
    try:
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            raise ValueError('invalid_box_schema')
        x1, y1, x2, y2 = [_number(v) for v in box]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError('box_out_of_bounds_or_non_positive_area')
    except ValueError as exc:
        return _unknown('invalid_box:' + str(exc))
    polygon = _rational(validation['points_px'])
    rational_box = tuple(Fraction(v) for v in (x1, y1, x2, y2))
    try:
        exact_area = sum((_clip_triangle(triangle, rational_box)
                          for triangle in _triangulate(polygon)), Fraction(0))
        numeric_area = _float_area(exact_area)
    except (ValueError, ZeroDivisionError) as exc:
        return _unknown('geometry_computation_failed:' + str(exc))
    box_area = (rational_box[2] - rational_box[0]) * (rational_box[3] - rational_box[1])
    touch = _boundary_contact(polygon, rational_box)
    crossing = 0 < exact_area < box_area
    if exact_area > 0:
        decision, reason = 'retain_candidate', 'positive_area_intersection'
    elif touch:
        decision, reason = 'review_boundary', 'zero_area_boundary_touch'
    else:
        decision, reason = 'suppress_outside_candidate', 'no_intersection'
    return {'intersection_area_px2': numeric_area, 'boundary_touch': touch,
            'boundary_crossing': crossing, 'decision': decision, 'reason': reason}


def safe_rate(n, d):
    """Return a rate with its supports; a zero denominator is never smoothed."""
    result = {'value': None, 'numerator': n, 'denominator': d, 'reason': None}
    try:
        numerator, denominator = _number(n), _number(d)
        if numerator < 0 or denominator < 0:
            raise ValueError('negative_count')
    except ValueError:
        result['reason'] = 'invalid_counts'
        return result
    if denominator == 0:
        result['reason'] = 'no_support'
    else:
        result['value'] = numerator / denominator
    return result


def target_metrics(labels, predictions, routing):
    """Fail closed on pending labels; this pilot does not implement GT matching.

    Empty pending objects are NOT negative frames. All four target metrics stay
    null until annotations are complete AND a separately verified global
    one-to-one evaluator is implemented. Predictions/routing cannot fabricate
    missing GT denominators. Complete data returns evaluator_not_implemented.
    """
    if isinstance(labels, dict):
        labels = [labels]
    complete = (isinstance(labels, (list, tuple)) and bool(labels)
                and all(isinstance(row, dict)
                        and row.get('annotation_complete') is True
                        and row.get('annotation_status') == 'complete'
                        and isinstance(row.get('objects'), list)
                        for row in labels))
    reason = 'evaluator_not_implemented' if complete else 'annotations_incomplete'
    return {'status': 'blocked', 'reason': reason,
            'metrics': {name: {'value': None, 'numerator': None,
                               'denominator': None, 'reason': reason}
                        for name in TARGET_METRIC_NAMES}}
