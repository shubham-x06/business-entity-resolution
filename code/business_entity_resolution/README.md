# Business Entity Resolution — Reproduction Guide

This guide details how to set up the environment, reproduce the candidate blocking stage against the dataset, inspect outputs, and run the test suite.

> **Status Note**: Candidate generation (blocking), data loading/I/O, text normalization, and macro $F_{0.5}$ metric evaluation are fully implemented, verified, and benchmarked on the full real dataset. Downstream pairwise feature extraction, LightGBM model training, and end-to-end test inference are currently in progress or staged on feature branches (see [Not Yet Implemented](#not-yet-implemented--in-progress-stages) below).

---

## 1. Environment Setup

The pipeline requires **Python 3.10+** (Python 3.11 recommended). It is managed with [`uv`](https://docs.astral.sh/uv/) for fast, deterministic dependency resolution.

### Using `uv` (Recommended)

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate       # On Linux / macOS
# .venv\Scripts\activate        # On Windows (PowerShell / Command Prompt)

# Install dependencies and editable package
uv pip install -r code/business_entity_resolution/requirements.txt
uv pip install -e code/business_entity_resolution
```

### Using standard `pip`

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# .venv\Scripts\activate        # On Windows (PowerShell / Command Prompt)

# Install dependencies and editable package
pip install -r code/business_entity_resolution/requirements.txt
pip install -e code/business_entity_resolution
```

---

## 2. Expected Directory Structure

The repository root must be structured as follows:

```text
business-entity-resolution/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv           # Reference S1 train records (2,206,821 rows)
│   │   ├── train_source2.tsv           # Source 2 train records (5,057,750 rows)
│   │   ├── train_source3.tsv           # Source 3 train records (5,262,469 rows)
│   │   └── train_ground_truth.tsv      # S1 -> matched S2/S3 IDs (7,638,365 true pairs)
│   └── test/
│       ├── test_source1.tsv            # Reference S1 test records
│       ├── test_source2.tsv            # Source 2 test records
│       └── test_source3.tsv            # Source 3 test records
├── checkpoints/                        # Incremental partition checkpoints
│   ├── candidates_india.tsv
│   └── candidates_us.tsv
├── output/                             # Generated submission artifacts
│   ├── candidate_pairs.tsv             # Blocking candidate set (Stage 1 output)
│   └── matching_results.tsv            # Final entity match predictions (Leaderboard)
├── experiments/                        # Experiment logs and tuning records
├── models/                             # Trained LightGBM model artifacts
├── code/
│   └── business_entity_resolution/
│       ├── configs/
│       │   ├── paths.yaml              # Dataset and output path mappings
│       │   ├── blocking.yaml           # Blocking hyperparameters and caps
│       │   └── model.yaml              # LightGBM classifier & threshold settings
│       ├── scripts/
│       │   ├── run_train.py            # Training & blocking execution CLI
│       │   └── run_infer.py            # Test-set inference CLI
│       ├── src/
│       │   └── business_entity_resolution/
│       │       ├── __init__.py
│       │       ├── blocking.py         # Multi-signal inverted index blocking
│       │       ├── io_utils.py         # TSV validation and safe file I/O
│       │       ├── normalize.py        # Unicode-safe text & address normalization
│       │       ├── scoring.py          # Exact competition macro F_0.5 evaluator
│       │       ├── features.py         # [In Progress] Pairwise feature extraction
│       │       ├── model.py            # [In Progress] LightGBM model training
│       │       ├── grouping.py         # [Branch: feature/grouping] Match clustering
│       │       └── pipeline.py         # [Branch: feature/pipeline-skeleton] Orchestration
│       ├── requirements.txt
│       └── README.md
└── tests/                              # Pytest test suite
```

---

## 3. Reproducing Candidate Blocking (Implemented)

The candidate blocking stage shrinks the pairwise comparison space from $O(N_1 \times (N_2 + N_3)) \approx 2.27 \times 10^{13}$ pairs down to a high-recall candidate pool of at most $K=200$ candidates per entity.

### Fast Prototyping / Dry Run

To verify the blocking implementation without running the full dataset, supply the `--sample` flag:

```bash
# Subsample first 1,000 Source 1 records
python code/business_entity_resolution/scripts/run_train.py --stage blocking --sample 1000
```

### Full-Scale Blocking Run

To run full blocking across all 2.2M Source 1 records and 10.3M reference records:

```bash
python code/business_entity_resolution/scripts/run_train.py --stage blocking
```

### Runtime and Checkpointing

- **Expected Runtime**: **~5.3 hours** aggregate execution time on an 8-core CPU system (~102.1 queries/sec aggregate throughput).
- **Fault-Tolerant Checkpointing**: The script writes intermediate partition checkpoints to `checkpoints/candidates_india.tsv` and `checkpoints/candidates_us.tsv` before combining them into `output/candidate_pairs.tsv`. If interrupted, completed partitions are preserved.
- **Output Artifact**: Generates `output/candidate_pairs.tsv` (440,091,505 candidate pairs total, mean 199.4 candidates per entity).

### Empirical Benchmark Results

Evaluated against the complete training ground truth (**7,638,365 true pairs** across 2,206,821 Source 1 entities):

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Reduction Ratio** | **99.9981%** | Comparison space reduced by $>52,000\times$ |
| **Recall@50** | **88.79%** | 6,781,778 true pairs retained |
| **Recall@100** | **90.95%** | 6,946,976 true pairs retained |
| **Recall@150** | **91.73%** | 7,006,359 true pairs retained |
| **Recall@200** | **92.20%** | 7,042,261 true pairs retained |
| **Throughput** | **102.1 q/s** | End-to-end query indexing and retrieval |

---

## 4. Not Yet Implemented / In-Progress Stages

To maintain transparency on reproducibility, the following pipeline components are currently stubs or undergoing branch staging:

1. **Stage 2: Feature Engineering (`features.py`)**  
   - Computes pairwise fuzzy and lexical features (Token Jaccard, Normalized Levenshtein, RapidFuzz token set/sort ratios, numeric address token overlap, country match flags).  
   - *Status*: Active development.

2. **Stage 3: Classifier Training (`model.py`)**  
   - LightGBM binary pairwise classifier trained to score candidate pairs with balanced class weights.  
   - *Status*: Active development.

3. **Stage 4: Post-Processing & Threshold Optimization (`grouping.py`)**  
   - Out-of-fold threshold grid search specifically optimizing the macro $F_{0.5}$ metric (rewarding true singletons at 1.0), and grouping many-to-many matches into clusters.  
   - *Status*: Complete and tested on branch `feature/grouping`; pending merge into `main`.

4. **Pipeline Orchestration & Test Inference (`pipeline.py`, `run_infer.py`)**  
   - End-to-end stage runner with cascade skipping and inference script producing `output/matching_results.tsv`.  
   - *Status*: Complete and tested on branch `feature/pipeline-skeleton`; pending merge into `main`.

---

## 5. Running the Test Suite

All unit tests and regression checks are run using `pytest`:

```bash
# Run full test suite with verbose output
pytest tests/ -v

# Or using uv directly:
uv run pytest tests/ -v
```

### Covered Test Modules

- `tests/test_scaffold.py`: Verifies module imports, package structure, CLI help flags, and YAML config syntax.
- `tests/test_normalize.py`: Verifies legal suffix canonicalization (Corp, Pvt, Ltd, Inc), street abbreviations (Rd, St, Ave, Blvd), ampersand expansion, Devanagari Hindi handling, and open-set country normalization (e.g. France).
- `tests/test_io_format.py`: Verifies byte-exact TSV roundtrip for `matching_results.tsv` and `candidate_pairs.tsv`, header formatting, missing-file handling, and ground-truth validation.
- `tests/test_scoring.py`: Verifies precision-weighted $F_{0.5}$ calculation, macro averaging, singleton edge cases (empty match predictions scored at 1.0 vs. false positive penalty 0.0), and asymmetry validation.
- `tests/test_blocking.py`: Verifies candidate blocking partition checkpointing and incremental writes.
