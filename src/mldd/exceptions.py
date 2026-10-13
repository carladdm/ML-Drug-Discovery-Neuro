"""Exception hierarchy for mldd.

Raise these domain-specific errors instead of bare ``Exception`` so callers
can react precisely to failure modes.
"""

from __future__ import annotations


class MLDDException(Exception):
    """Base class for every mldd error."""


class ConfigError(MLDDException):
    """Invalid or missing configuration."""


class IngestionError(MLDDException):
    """ChEMBL ingestion failed (network error, empty result, rate-limit)."""


class InvalidSMILESError(MLDDException):
    """A SMILES string could not be parsed by RDKit."""


class FeaturizationError(MLDDException):
    """Descriptor or fingerprint computation failed."""


class TrainingError(MLDDException):
    """Model training or hyperparameter-tuning failed."""


class ModelNotFoundError(MLDDException):
    """The requested model artifact does not exist on disk or in the registry."""
