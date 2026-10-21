"""Evaluation against public datasets with 3D ground truth. See ADR-0004."""

from unproject.ground_truth.artifacts import (
    EvaluationDescriptor,
    read_descriptor,
    staged_evaluation,
    write_descriptor,
)
from unproject.ground_truth.config import (
    PROTOCOLS,
    GroundTruthConfig,
    ProtocolSpec,
    load_ground_truth_config,
)
from unproject.ground_truth.dataset import (
    CANONICAL_CONVENTION,
    DatasetRecord,
    GroundTruthDataset,
    GroundTruthError,
    GroundTruthPerson,
    validate_record,
)
from unproject.ground_truth.inventory import Inventory, inventory, read_subset_manifest
from unproject.ground_truth.mapping import JointMapping, apply_mapping, load_mapping

__all__ = [
    "CANONICAL_CONVENTION",
    "PROTOCOLS",
    "DatasetRecord",
    "EvaluationDescriptor",
    "GroundTruthConfig",
    "GroundTruthDataset",
    "GroundTruthError",
    "GroundTruthPerson",
    "Inventory",
    "JointMapping",
    "ProtocolSpec",
    "apply_mapping",
    "inventory",
    "load_ground_truth_config",
    "load_mapping",
    "read_descriptor",
    "read_subset_manifest",
    "staged_evaluation",
    "validate_record",
    "write_descriptor",
]
