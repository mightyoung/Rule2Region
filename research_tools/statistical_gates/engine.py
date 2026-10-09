"""Small deterministic implementations of the preregistered statistical gates.

The engine intentionally accepts already blinded, post-exclusion summaries.  It
does not read predictions or estimate effects from a pilot.
"""
from __future__ import annotations
import hashlib, json, math
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any, Mapping, Sequence
import numpy as np

CAL_ACC4 = "CAL-ACC4"
CAL_INT2 = "CAL-INT2"

def _as_cameras(values: Any) -> list[list[float]]:
    if isinstance(values, Mapping): values = list(values.values())
    return [list(map(float, v)) for v in values]

def equal_camera_estimand(values: Any) -> float:
    """Equal-camera mean of equal-source means (windows are pre-averaged)."""
    cams = _as_cameras(values)
    if not cams or any(not c for c in cams): raise ValueError("non-empty cameras required")
    return float(np.mean([np.mean(c) for c in cams]))

def hierarchical_bootstrap(values: Any, draws: int = 10000, seed: int = 20260728) -> np.ndarray:
    """Camera-first paired bootstrap; each sampled camera occurrence resamples sources."""
    cams = _as_cameras(values)
    if not cams or any(not c for c in cams): raise ValueError("non-empty cameras required")
    rng = np.random.default_rng(seed); c = len(cams)
    out = np.empty(draws, dtype=float)
    for i in range(draws):
        ci = rng.integers(0, c, size=c)
        out[i] = np.mean([np.mean(np.asarray(cams[j])[rng.integers(0, len(cams[j]), size=len(cams[j]))]) for j in ci])
    return out

def sign_flip_pvalue(camera_values: Sequence[float], seed: int = 20260801, draws: int = 100000) -> float:
    """Two-sided camera-level sign flip, exact through 2**C <= 100,000."""
    x = np.asarray(camera_values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)): raise ValueError("finite camera values required")
    observed = abs(float(np.mean(x))); n = len(x)
    if 2 ** n <= draws:
        stats = np.abs(_exact_sign_matrix(n) @ x / n)
        return float(np.mean(stats >= observed - 1e-15))
    rng = np.random.default_rng(seed); signs = rng.choice((-1.0, 1.0), size=(draws, n))
    extreme = int(np.count_nonzero(np.abs(signs @ x / n) >= observed - 1e-15))
    return (extreme + 1) / (draws + 1)

def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or len(p) == 0 or np.any((p < 0) | (p > 1)): raise ValueError("p-values must lie in [0,1]")
    order = np.argsort(p); adj = np.empty(len(p)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx]); adj[idx] = min(1.0, running)
    return adj.tolist()

def wilson_interval(rejections: int, simulations: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if simulations <= 0 or not (0 <= rejections <= simulations): raise ValueError("invalid binomial counts")
    n, phat = simulations, rejections / simulations; den = 1 + z*z/n
    centre = (phat + z*z/(2*n))/den; half = z*math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/den
    return (max(0.0, centre-half), min(1.0, centre+half))

@dataclass(frozen=True)
class CalibrationResult:
    artifact_id: str; seed: int; simulations: int; rejections: int; failures: int
    fwer: float; wilson95: tuple[float, float]; verdict: str; receipt: str
    dgp_sha256: str; covariance_sha256: str; config_sha256: str; result_sha256: str
    evidence_level: str = "FIXTURE_VERIFIED"
    def to_dict(self): return self.__dict__.copy()

def sha256_seal(*objects: Any) -> str:
    payload = json.dumps(objects, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()

def _joint_null_fixture(contrasts: int, cameras: int = 14, rho: float = 0.10) -> tuple[dict[str, Any], np.ndarray]:
    """Return a symmetric correlated camera-level global-null DGP."""
    if cameras < 8:
        raise ValueError("calibration fixture requires at least eight cameras")
    if not (-1 / (contrasts - 1) < rho < 1):
        raise ValueError("invalid equicorrelation")
    covariance = np.full((contrasts, contrasts), rho, dtype=float)
    np.fill_diagonal(covariance, 1.0)
    dgp = {
        "name": "joint_gaussian_camera_null_v1",
        "cameras": cameras,
        "contrasts": contrasts,
        "mean": [0.0] * contrasts,
        "rho": rho,
        "sampling_unit": "camera",
        "test": "two_sided_exact_camera_sign_flip",
        "multiplicity": "holm",
    }
    return dgp, covariance

def run_calibration(artifact_id: str, contrasts: int, seed: int, simulations: int = 20000,
                    *, statistic=None, failure_indices: Sequence[int] = ()) -> CalibrationResult:
    """Run a deterministic joint camera-level global-null calibration.

    The default path samples correlated camera-level contrasts, executes the
    exact sign-flip test for each contrast, applies Holm across the registered
    family, and records whether any true null was rejected. A caller-supplied
    statistic is reserved for engine failure/boundary fixtures.
    """
    if artifact_id not in (CAL_ACC4, CAL_INT2): raise ValueError("unknown calibration family")
    expected = 4 if artifact_id == CAL_ACC4 else 2
    if contrasts != expected: raise ValueError(f"{artifact_id} requires exactly {expected} contrasts")
    if simulations != 20000: raise ValueError("calibration requires exactly 20,000 simulations")
    dgp, covariance = _joint_null_fixture(contrasts)
    config = {
        "artifact_id": artifact_id,
        "contrasts": contrasts,
        "seed": seed,
        "simulations": simulations,
        "alpha": 0.05,
        "family_rejection": "any_holm_adjusted_p_le_alpha",
    }
    bad = set(failure_indices); rng = np.random.default_rng(seed); rejected = failures = 0
    for i in range(simulations):
        try:
            if i in bad: raise FloatingPointError("fixture failure")
            if statistic is None:
                camera_contrasts = rng.multivariate_normal(
                    mean=np.zeros(contrasts), cov=covariance, size=dgp["cameras"]
                )
                pvalues = [
                    sign_flip_pvalue(camera_contrasts[:, j])
                    for j in range(contrasts)
                ]
                hit = min(holm_adjust(pvalues)) <= config["alpha"]
            else:
                raw_hit = statistic(rng, i)
                if not isinstance(raw_hit, (bool, np.bool_)):
                    raise FloatingPointError("statistic must return a finite boolean")
                hit = bool(raw_hit)
            rejected += int(hit)
        except Exception: failures += 1
    fwer = rejected / simulations; ci = wilson_interval(rejected, simulations)
    verdict = "PASS" if .04 <= fwer <= .06 and ci[0] <= .05 <= ci[1] and failures / simulations <= .01 else f"{artifact_id.replace('-', '_')}_INVALID"
    dgp_sha256 = sha256_seal(dgp)
    covariance_sha256 = sha256_seal(covariance.tolist())
    config_sha256 = sha256_seal(config)
    result_payload = {
        "rejections": rejected,
        "failures": failures,
        "fwer": fwer,
        "wilson95": ci,
        "verdict": verdict,
    }
    result_sha256 = sha256_seal(result_payload)
    receipt = sha256_seal(dgp_sha256, covariance_sha256, config_sha256, result_sha256)
    return CalibrationResult(
        artifact_id, seed, simulations, rejected, failures, fwer, ci, verdict,
        receipt, dgp_sha256, covariance_sha256, config_sha256, result_sha256,
    )

@dataclass(frozen=True)
class FormalStructure:
    C_formal: int; V_c: tuple[int, ...]; W_cv: tuple[tuple[int, ...], ...]; sample_hash: str

def validate_power_inputs(structure: FormalStructure, pilot: Mapping[str, Any]) -> None:
    """Fail closed unless exact post-exclusion formal structure and blinded pilot exist."""
    if structure.C_formal != len(structure.V_c) or structure.C_formal != len(structure.W_cv): raise ValueError("inconsistent formal structure")
    if structure.C_formal < 8 or sum(structure.V_c) < 30 or any(v < 2 for v in structure.V_c): raise ValueError("formal floor 8/30/2 failed")
    if any(len(ws) != v or any(w <= 0 for w in ws) for v, ws in zip(structure.V_c, structure.W_cv)): raise ValueError("invalid window structure")
    if not structure.sample_hash: raise ValueError("formal sample hash required")
    if not pilot.get("blinded", False) or "variance" not in pilot or "icc" not in pilot: raise ValueError("blinded-pilot variance/ICC required")
    if any(k in pilot for k in ("effect", "method_effect", "delta")): raise ValueError("pilot method effect forbidden")
@lru_cache(maxsize=None)
def _exact_sign_matrix(cameras: int) -> np.ndarray:
    return np.asarray(list(product((-1.0, 1.0), repeat=cameras)), dtype=float)
