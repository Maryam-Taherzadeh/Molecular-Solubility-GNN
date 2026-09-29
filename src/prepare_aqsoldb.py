# ============================================================
# ============================================================
# PREPARE AQSOLDB FOR GCN AND GAT TRAINING
#
# This script:
# 1. Loads the original AqSolDB CSV file
# 2. Selects the SMILES and solubility columns
# 3. Converts solubility values to numeric logS values
# 4. Validates molecular structures using RDKit
# 5. Removes disconnected salt counterions by retaining the
#    largest organic covalent fragment
# 6. Removes compounds without carbon, invalid structures,
#    extremely small molecules, and unsupported elements
# 7. Combines duplicate parent molecules and averages repeated
#    solubility measurements
# 8. Records which molecules originally contained salts or
#    multiple disconnected fragments
# 9. Creates reproducible random train, validation, and test splits
# 10. Saves the cleaned, removed, and split datasets
#
# Important:
# - Salt-containing rows are not automatically deleted.
# - For multi-fragment structures, the largest organic fragment
#   is retained and smaller counterions are removed.
# - The final cleaned SMILES contain only one connected fragment.
# - Compounds without carbon are removed because they are outside
#   the current organic, drug-like modeling domain.
# - Duplicate handling is performed after salt stripping.
# - No arbitrary logS range filter is applied.
# - The same salt-stripped dataset is used for both GCN and GAT.
# - The current train, validation, and test files use a random split.
# - A separate scaffold split can be created later from the same
#   cleaned dataset for a stricter generalization evaluation.
# ============================================================
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path
import random

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from sklearn.model_selection import train_test_split


# ============================================================
# 2. GENERAL CONFIGURATION
# ============================================================

# Hide repetitive RDKit warning messages.
# Invalid structures are still detected and removed.
RDLogger.DisableLog("rdApp.*")


# Fixed random seed for reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# Expected project structure:
#
# project/
# ├── data/
# │   ├── raw/
# │   │   └── aqsoldb.csv
# │   └── processed/
# └── src/
#     └── data/
#         └── prepare_aqsoldb.py
#
# parents[2] gives the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Raw AqSolDB path
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aqsoldb.csv"
)


# Processed-data directory
PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. OUTPUT FILE PATHS
# ============================================================

CLEANED_DATA_PATH = (
    PROCESSED_DIR
    / "aqsoldb_cleaned.csv"
)

REMOVED_STRUCTURES_PATH = (
    PROCESSED_DIR
    / "aqsoldb_removed_structures.csv"
)

TRAIN_DATA_PATH = (
    PROCESSED_DIR
    / "aqsoldb_train.csv"
)

VALIDATION_DATA_PATH = (
    PROCESSED_DIR
    / "aqsoldb_validation.csv"
)

TEST_DATA_PATH = (
    PROCESSED_DIR
    / "aqsoldb_test.csv"
)

SPLIT_SUMMARY_PATH = (
    PROCESSED_DIR
    / "aqsoldb_split_summary.csv"
)


# ============================================================
# 4. LOAD THE RAW AQSOLDB DATASET
# ============================================================

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
        "The raw AqSolDB dataset was not found.\n"
        f"Expected location: {RAW_DATA_PATH}"
    )


raw_df = pd.read_csv(
    RAW_DATA_PATH
)


print("\nRaw AqSolDB dataset")
print("=" * 60)

print(
    "Dataset path:",
    RAW_DATA_PATH,
)

print(
    "Original shape:",
    raw_df.shape,
)

print(
    "Available columns:",
    raw_df.columns.tolist(),
)


# Required columns for this model
required_columns = {
    "SMILES",
    "Solubility",
}


missing_columns = required_columns.difference(
    raw_df.columns
)


if missing_columns:
    raise ValueError(
        "The following required columns are missing: "
        f"{sorted(missing_columns)}"
    )


# ============================================================
# 5. SELECT THE REQUIRED COLUMNS
# ============================================================

working_df = raw_df[
    [
        "SMILES",
        "Solubility",
    ]
].copy()


working_df = working_df.rename(
    columns={
        "SMILES": "original_smiles",
        "Solubility": "logS",
    }
)


starting_rows = len(
    working_df
)


# Convert solubility values to numeric.
# Invalid text values become NaN.
working_df["logS"] = pd.to_numeric(
    working_df["logS"],
    errors="coerce",
)


missing_target_count = int(
    working_df["logS"].isna().sum()
)


# ============================================================
# 6. DEFINE ELEMENTS SUPPORTED BY THE CURRENT MODEL
# ============================================================

# These elements must also be supported by the graph
# feature-generation code used by the GCN and GAT.
ALLOWED_ELEMENTS = {
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
}


# ============================================================
# 7. STANDARDIZE MOLECULAR STRUCTURES
# ============================================================

def standardize_structure(smiles):
    """
    Validate a SMILES value and retain the largest organic
    covalent fragment.

    Salt-containing rows are not automatically deleted.
    Instead, the parent organic structure is retained while
    smaller disconnected fragments, such as counterions, are
    removed.

    Returns
    -------
    dict
        parent_smiles:
            Canonical SMILES of the retained parent molecule.

        salt_removed:
            True when the original structure contained multiple
            disconnected fragments.

        original_fragment_count:
            Number of disconnected components in the original
            molecular structure.

        standardization_reason:
            Result of the standardization operation.
    """

    result = {
        "parent_smiles": None,
        "salt_removed": False,
        "original_fragment_count": 0,
        "standardization_reason": None,
    }

    # Reject missing SMILES values
    if pd.isna(smiles):
        result["standardization_reason"] = "missing_smiles"
        return result

    # Convert input to text and remove extra spaces
    smiles = str(smiles).strip()

    # Reject empty SMILES strings
    if not smiles:
        result["standardization_reason"] = "empty_smiles"
        return result

    try:
        # Parse the original molecular structure
        molecule = Chem.MolFromSmiles(
            smiles
        )

        if molecule is None:
            result["standardization_reason"] = "invalid_smiles"
            return result

        # Count disconnected fragments in the original SMILES
        fragment_count = len(
            Chem.GetMolFrags(
                molecule
            )
        )

        result["original_fragment_count"] = fragment_count
        result["salt_removed"] = fragment_count > 1

        # Retain the largest organic fragment.
        #
        # Some RDKit versions do not accept preferOrganic=True,
        # so a fallback is included.
        try:
            parent_molecule = rdMolStandardize.FragmentParent(
                molecule,
                preferOrganic=True,
            )

        except TypeError:
            parent_molecule = rdMolStandardize.FragmentParent(
                molecule
            )

        if (
            parent_molecule is None
            or parent_molecule.GetNumAtoms() == 0
        ):
            result["standardization_reason"] = (
                "no_valid_parent_fragment"
            )
            return result

        # Require at least one carbon atom
        contains_carbon = any(
            atom.GetAtomicNum() == 6
            for atom in parent_molecule.GetAtoms()
        )

        if not contains_carbon:
            result["standardization_reason"] = "no_carbon"
            return result

        # Convert the retained parent to canonical SMILES
        parent_smiles = Chem.MolToSmiles(
            parent_molecule,
            canonical=True,
            isomericSmiles=True,
        )

        if not parent_smiles:
            result["standardization_reason"] = (
                "empty_parent_smiles"
            )
            return result

        result["parent_smiles"] = parent_smiles
        result["standardization_reason"] = "standardized"

        return result

    except Exception as error:
        result["standardization_reason"] = (
            f"standardization_error:{type(error).__name__}"
        )
        return result


# Apply molecular standardization to every row
standardization_results = working_df[
    "original_smiles"
].apply(
    standardize_structure
)


# Extract standardization outputs
working_df["smiles"] = standardization_results.apply(
    lambda result: result["parent_smiles"]
)

working_df["salt_removed"] = standardization_results.apply(
    lambda result: result["salt_removed"]
)

working_df["original_fragment_count"] = (
    standardization_results.apply(
        lambda result: result["original_fragment_count"]
    )
)

working_df["standardization_reason"] = (
    standardization_results.apply(
        lambda result: result["standardization_reason"]
    )
)


invalid_or_removed_smiles_count = int(
    working_df["smiles"].isna().sum()
)


# ============================================================
# 8. STORE STRUCTURES REMOVED DURING STANDARDIZATION
# ============================================================

invalid_target_mask = (
    working_df["logS"].isna()
    | ~np.isfinite(
        working_df["logS"]
    )
)

invalid_structure_mask = (
    working_df["smiles"].isna()
)


initial_removed_mask = (
    invalid_target_mask
    | invalid_structure_mask
)


initial_removed_df = working_df[
    initial_removed_mask
].copy()


# Assign target-related removal reasons first
initial_removed_df["filter_reason"] = pd.Series(
    pd.NA,
    index=initial_removed_df.index,
    dtype="string",
)

initial_removed_df.loc[
    initial_removed_df["logS"].isna(),
    "filter_reason",
] = "missing_or_nonnumeric_target"


nonfinite_target_mask = (
    initial_removed_df["logS"].notna()
    & ~np.isfinite(
        initial_removed_df["logS"]
    )
)

initial_removed_df.loc[
    nonfinite_target_mask,
    "filter_reason",
] = "nonfinite_target"


# Use molecular standardization reasons for remaining rows
missing_reason_mask = (
    initial_removed_df["filter_reason"].isna()
)

initial_removed_df.loc[
    missing_reason_mask,
    "filter_reason",
] = initial_removed_df.loc[
    missing_reason_mask,
    "standardization_reason",
]


# Keep only rows with valid structures and finite targets
working_df = working_df[
    ~initial_removed_mask
].copy()


working_df = working_df.reset_index(
    drop=True
)


valid_rows_before_structure_filter = len(
    working_df
)


# ============================================================
# 9. FILTER STRUCTURES UNSUITABLE FOR THE CURRENT MODEL
# ============================================================

def check_parent_structure(smiles):
    """
    Check whether a standardized parent molecule is suitable
    for the current GCN and GAT feature system.

    Returns
    -------
    suitable : bool
        True when the molecule can be used.

    reason : str
        Reason for keeping or removing the molecule.
    """

    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return False, "invalid_parent_smiles"

    # Confirm that the retained structure contains carbon
    contains_carbon = any(
        atom.GetAtomicNum() == 6
        for atom in molecule.GetAtoms()
    )

    if not contains_carbon:
        return False, "no_carbon"

    # Remove extremely small parent structures
    if molecule.GetNumHeavyAtoms() < 3:
        return False, "fewer_than_3_heavy_atoms"

    # FragmentParent should normally produce one fragment
    fragment_count = len(
        Chem.GetMolFrags(
            molecule
        )
    )

    if fragment_count != 1:
        return False, "parent_still_has_multiple_fragments"

    # Collect all element symbols in the retained parent
    molecule_elements = {
        atom.GetSymbol()
        for atom in molecule.GetAtoms()
    }

    # Check whether all elements are supported
    if not molecule_elements.issubset(
        ALLOWED_ELEMENTS
    ):
        unsupported_elements = sorted(
            molecule_elements.difference(
                ALLOWED_ELEMENTS
            )
        )

        return (
            False,
            "unsupported_element:"
            + ",".join(unsupported_elements),
        )

    return True, "kept"


# Apply the structure filter
structure_results = working_df[
    "smiles"
].apply(
    check_parent_structure
)


working_df["suitable_for_model"] = (
    structure_results.apply(
        lambda result: result[0]
    )
)


working_df["filter_reason"] = (
    structure_results.apply(
        lambda result: result[1]
    )
)


# Save structures rejected after parent standardization
structure_removed_df = working_df[
    ~working_df["suitable_for_model"]
].copy()


# Keep valid structures only
working_df = working_df[
    working_df["suitable_for_model"]
].copy()


# Remove temporary processing columns safely.
# Only columns that currently exist are removed.
temporary_columns = [
    "suitable_for_model",
    "filter_reason",
    "standardization_reason",
]


columns_to_remove = [
    column
    for column in temporary_columns
    if column in working_df.columns
]


working_df = working_df.drop(
    columns=columns_to_remove
).reset_index(
    drop=True
)


valid_rows_before_deduplication = len(
    working_df
)


# ============================================================
# 10. COMBINE DUPLICATE PARENT MOLECULES
# ============================================================

# Duplicate handling happens after salt stripping because
# different salts may map to the same parent molecule.
duplicate_rows = int(
    working_df.duplicated(
        subset="smiles",
        keep=False,
    ).sum()
)


clean_df = (
    working_df.groupby(
        "smiles",
        as_index=False,
    )
    .agg(
        logS=(
            "logS",
            "mean",
        ),
        logS_std=(
            "logS",
            "std",
        ),
        measurement_count=(
            "logS",
            "size",
        ),
        salt_removed=(
            "salt_removed",
            "max",
        ),
        salt_record_count=(
            "salt_removed",
            "sum",
        ),
        maximum_original_fragment_count=(
            "original_fragment_count",
            "max",
        ),
    )
)


# A single measurement has no sample standard deviation.
# Replace NaN with zero.
clean_df["logS_std"] = clean_df[
    "logS_std"
].fillna(0.0)


# Set consistent data types
clean_df["salt_removed"] = clean_df[
    "salt_removed"
].astype(bool)


clean_df["salt_record_count"] = clean_df[
    "salt_record_count"
].astype(int)


clean_df["measurement_count"] = clean_df[
    "measurement_count"
].astype(int)


clean_df[
    "maximum_original_fragment_count"
] = clean_df[
    "maximum_original_fragment_count"
].astype(int)


clean_df = clean_df.reset_index(
    drop=True
)


# ============================================================
# 11. COMBINE ALL REMOVED STRUCTURES
# ============================================================

removed_columns = [
    "original_smiles",
    "logS",
    "smiles",
    "salt_removed",
    "original_fragment_count",
    "filter_reason",
]


# Ensure both removed-data dataframes have the same columns
for column in removed_columns:
    if column not in initial_removed_df.columns:
        initial_removed_df[column] = np.nan

    if column not in structure_removed_df.columns:
        structure_removed_df[column] = np.nan


removed_structures_df = pd.concat(
    [
        initial_removed_df[removed_columns],
        structure_removed_df[removed_columns],
    ],
    ignore_index=True,
)


removed_structures_df = (
    removed_structures_df.reset_index(
        drop=True
    )
)


# ============================================================
# 12. FINAL DATA VALIDATION
# ============================================================

remaining_missing_smiles = int(
    clean_df["smiles"].isna().sum()
)


remaining_missing_targets = int(
    clean_df["logS"].isna().sum()
)


remaining_nonfinite_targets = int(
    (
        ~np.isfinite(
            clean_df["logS"]
        )
    ).sum()
)


remaining_duplicate_smiles = int(
    clean_df["smiles"].duplicated().sum()
)


remaining_invalid_smiles = int(
    clean_df["smiles"].apply(
        lambda smiles: (
            Chem.MolFromSmiles(smiles) is None
        )
    ).sum()
)


def has_multiple_fragments(smiles):
    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return True

    return len(
        Chem.GetMolFrags(
            molecule
        )
    ) != 1


remaining_multifragment_smiles = int(
    clean_df["smiles"].apply(
        has_multiple_fragments
    ).sum()
)


def has_no_carbon(smiles):
    molecule = Chem.MolFromSmiles(
        smiles
    )

    if molecule is None:
        return True

    return not any(
        atom.GetAtomicNum() == 6
        for atom in molecule.GetAtoms()
    )


remaining_no_carbon = int(
    clean_df["smiles"].apply(
        has_no_carbon
    ).sum()
)


if remaining_missing_smiles != 0:
    raise ValueError(
        "Missing SMILES remain after cleaning."
    )


if remaining_missing_targets != 0:
    raise ValueError(
        "Missing logS values remain after cleaning."
    )


if remaining_nonfinite_targets != 0:
    raise ValueError(
        "Nonfinite logS values remain after cleaning."
    )


if remaining_duplicate_smiles != 0:
    raise ValueError(
        "Duplicate parent SMILES remain after cleaning."
    )


if remaining_invalid_smiles != 0:
    raise ValueError(
        "Invalid SMILES remain after cleaning."
    )


if remaining_multifragment_smiles != 0:
    raise ValueError(
        "Multiple-fragment parent structures remain."
    )


if remaining_no_carbon != 0:
    raise ValueError(
        "Carbon-free structures remain after cleaning."
    )


# ============================================================
# 13. PRINT THE CLEANING SUMMARY
# ============================================================

print("\nCleaning summary")
print("=" * 60)

print(
    "Starting rows:",
    starting_rows,
)

print(
    "Missing or nonnumeric targets:",
    missing_target_count,
)

print(
    "Invalid or removed molecular structures:",
    invalid_or_removed_smiles_count,
)

print(
    "Valid rows before structure filtering:",
    valid_rows_before_structure_filter,
)

print(
    "Valid rows before deduplication:",
    valid_rows_before_deduplication,
)

print(
    "Rows involved in duplicate parent structures:",
    duplicate_rows,
)

print(
    "Removed structures:",
    len(removed_structures_df),
)

print(
    "Final unique parent molecules:",
    len(clean_df),
)

print(
    "Final parent molecules originating from salts "
    "or multi-fragment structures:",
    int(
        clean_df["salt_removed"].sum()
    ),
)


print("\nRemoval reasons")
print("=" * 60)

if len(removed_structures_df) > 0:
    print(
        removed_structures_df[
            "filter_reason"
        ].value_counts(
            dropna=False
        )
    )

else:
    print(
        "No structures were removed."
    )


print("\nFinal validation")
print("=" * 60)

print(
    "Missing SMILES:",
    remaining_missing_smiles,
)

print(
    "Missing targets:",
    remaining_missing_targets,
)

print(
    "Nonfinite targets:",
    remaining_nonfinite_targets,
)

print(
    "Duplicate parent SMILES:",
    remaining_duplicate_smiles,
)

print(
    "Invalid SMILES:",
    remaining_invalid_smiles,
)

print(
    "Multiple-fragment parent structures:",
    remaining_multifragment_smiles,
)

print(
    "Carbon-free structures:",
    remaining_no_carbon,
)


print("\nCleaned logS summary")
print("=" * 60)

print(
    clean_df["logS"].describe()
)


print("\nMeasurement-count summary")
print("=" * 60)

print(
    clean_df["measurement_count"].describe()
)


print("\nFirst five cleaned molecules")
print("=" * 60)

print(
    clean_df.head()
)


# ============================================================
# 14. SAVE THE CLEANED AND REMOVED DATASETS
# ============================================================

clean_df.to_csv(
    CLEANED_DATA_PATH,
    index=False,
)


removed_structures_df.to_csv(
    REMOVED_STRUCTURES_PATH,
    index=False,
)


if not CLEANED_DATA_PATH.exists():
    raise FileNotFoundError(
        "The cleaned dataset was not saved:\n"
        f"{CLEANED_DATA_PATH}"
    )


if not REMOVED_STRUCTURES_PATH.exists():
    raise FileNotFoundError(
        "The removed-structures dataset was not saved:\n"
        f"{REMOVED_STRUCTURES_PATH}"
    )


# ============================================================
# 15. CREATE RANDOM TRAIN, VALIDATION, AND TEST SPLITS
# ============================================================

# Random split:
# - 80% training
# - 10% validation
# - 10% test
#
# This is useful as a baseline. A scaffold split should be used
# separately for a stricter molecular generalization test.

if len(clean_df) < 10:
    raise ValueError(
        "The cleaned dataset is too small to create reliable "
        "train, validation, and test splits."
    )


train_df, temporary_df = train_test_split(
    clean_df,
    test_size=0.20,
    random_state=SEED,
    shuffle=True,
)


validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=SEED,
    shuffle=True,
)


train_df = train_df.reset_index(
    drop=True
)

validation_df = validation_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)


# ============================================================
# 16. VERIFY THE DATASET SPLITS
# ============================================================

total_size = len(
    clean_df
)


split_total = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)


if split_total != total_size:
    raise ValueError(
        "The split sizes do not equal the cleaned dataset size."
    )


train_smiles = set(
    train_df["smiles"]
)

validation_smiles = set(
    validation_df["smiles"]
)

test_smiles = set(
    test_df["smiles"]
)


train_validation_overlap = (
    train_smiles.intersection(
        validation_smiles
    )
)


train_test_overlap = (
    train_smiles.intersection(
        test_smiles
    )
)


validation_test_overlap = (
    validation_smiles.intersection(
        test_smiles
    )
)


if train_validation_overlap:
    raise ValueError(
        "Train and validation datasets overlap."
    )


if train_test_overlap:
    raise ValueError(
        "Train and test datasets overlap."
    )


if validation_test_overlap:
    raise ValueError(
        "Validation and test datasets overlap."
    )


# ============================================================
# 17. CREATE THE SPLIT SUMMARY
# ============================================================

split_summary = pd.DataFrame(
    {
        "split": [
            "train",
            "validation",
            "test",
        ],
        "rows": [
            len(train_df),
            len(validation_df),
            len(test_df),
        ],
        "percentage": [
            len(train_df) / total_size,
            len(validation_df) / total_size,
            len(test_df) / total_size,
        ],
        "mean_logS": [
            train_df["logS"].mean(),
            validation_df["logS"].mean(),
            test_df["logS"].mean(),
        ],
        "std_logS": [
            train_df["logS"].std(),
            validation_df["logS"].std(),
            test_df["logS"].std(),
        ],
        "min_logS": [
            train_df["logS"].min(),
            validation_df["logS"].min(),
            test_df["logS"].min(),
        ],
        "max_logS": [
            train_df["logS"].max(),
            validation_df["logS"].max(),
            test_df["logS"].max(),
        ],
        "salt_parent_count": [
            int(
                train_df["salt_removed"].sum()
            ),
            int(
                validation_df["salt_removed"].sum()
            ),
            int(
                test_df["salt_removed"].sum()
            ),
        ],
    }
)


print("\nDataset split")
print("=" * 60)

print(
    f"Training:   {len(train_df):,} "
    f"({len(train_df) / total_size:.2%})"
)

print(
    f"Validation: {len(validation_df):,} "
    f"({len(validation_df) / total_size:.2%})"
)

print(
    f"Test:       {len(test_df):,} "
    f"({len(test_df) / total_size:.2%})"
)


print(
    "\nTrain-validation overlap:",
    len(train_validation_overlap),
)

print(
    "Train-test overlap:",
    len(train_test_overlap),
)

print(
    "Validation-test overlap:",
    len(validation_test_overlap),
)


print("\nSplit target summary")
print("=" * 60)

print(
    split_summary
)


# ============================================================
# 18. SAVE THE SPLIT DATASETS
# ============================================================

train_df.to_csv(
    TRAIN_DATA_PATH,
    index=False,
)


validation_df.to_csv(
    VALIDATION_DATA_PATH,
    index=False,
)


test_df.to_csv(
    TEST_DATA_PATH,
    index=False,
)


split_summary.to_csv(
    SPLIT_SUMMARY_PATH,
    index=False,
)


# ============================================================
# 19. VERIFY ALL SAVED FILES
# ============================================================

saved_paths = [
    CLEANED_DATA_PATH,
    REMOVED_STRUCTURES_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    TEST_DATA_PATH,
    SPLIT_SUMMARY_PATH,
]


print("\nSaved files")
print("=" * 60)


for path in saved_paths:
    if not path.exists():
        raise FileNotFoundError(
            "The following file was not saved:\n"
            f"{path}"
        )

    print(
        f"{path} "
        f"({path.stat().st_size:,} bytes)"
    )


print(
    "\nAqSolDB preparation completed successfully."
)