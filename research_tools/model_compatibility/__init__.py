"""Fail-closed fixed-revision Video-LMM compatibility checks."""

from .harness import FIXED_MODELS, validate_matrix, validate_receipt

__all__ = ["FIXED_MODELS", "validate_matrix", "validate_receipt"]
