# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TODO: fill in once Milestone 7-10 numbers are final]  
**Team Members:** [TODO: fill in once Milestone 7-10 numbers are final]  
**Submission Date:** [TODO: fill in once Milestone 7-10 numbers are final]

---

## 1. Executive Summary

We implement a scalable, production-grade business entity resolution pipeline designed for the Amazon ML Challenge 2026. The architecture couples Unicode-aware legal/street text normalization and country-partitioned multi-signal inverted index blocking (with asymmetric IDF frequency weighting) with a pairwise LightGBM classifier and a dynamic singleton-aware threshold sweep optimized specifically for the macro $F_{0.5}$ objective.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis revealed several critical domain-specific challenges across the 2.2M Source1 and 10.3M Source2/3 records:
- **Lexical and Syntactic Noise:** Extensive variations in legal designations (Corp vs. Corporation, Pvt vs. Private, Ltd vs. Limited), street abbreviations (Rd vs. Road, St vs. Street), punctuation, and ampersands.
- **Multilingual & Cross-Script Records:** Genuine Devanagari Hindi records coexisting with Latin-script transliterations in Indian partitions.
- **Extreme Class Imbalance & Scale:** Comparing 2.2M entities against 10.3M references creates a search space exceeding $2 \times 10^{13}$ pairs, requiring strictly sub-quadratic blocking.
- **Metric Asymmetry ($F_{0.5}$ and Singletons):** The $F_{0.5}$ metric penalizes false positive mergers twice as heavily as false negative recall loss, while singletons (entities with zero true matches) impose hard 1.0 vs. 0.0 boundary conditions.

### 2.2 Solution Strategy
- **Approach Type:** Country-Partitioned Inverted Index Blocking + Pairwise Gradient-Boosted Classification (LightGBM) + Macro $F_{0.5}$ Threshold Optimization.
- **Core Innovation:** Asymmetric token frequency capping coupled with continuous inverse document frequency (IDF) scoring in candidate blocking, ensuring discriminative corporate name words are never dropped while keeping generic street terms bounded, alongside a singleton-aware threshold selection strategy that maximizes the competition $F_{0.5}$ metric.

---

## 3. Candidate Generation (Blocking)

To reduce the $O(N_1 \times (N_2 + N_3)) \approx 2.2\text{M} \times 10.3\text{M} \approx 2.27 \times 10^{13}$ pairwise comparison space, we employ a high-throughput multi-signal blocking pipeline:

- **Country Partitioning**: Exact partitioning by country (`India`, `US`, and open-set `France` for test). Records from different countries are never cross-paired.
- **Multi-Signal Inverted Indices**:
  - Exact normalized name match (high weight = 10.0).
  - Informative name word tokens (length $\ge 3$, excluding standard business stop words) with IDF weighting.
  - Compound keys `(name_prefix_4, addr_token)` to pair entities with partial name overlap and shared location. For the US partition, 2-letter tokens are strictly filtered to an allowlist of valid US state abbreviations to avoid combinatorial explosion from generic 2-letter words.
  - Address token and bigram overlap with IDF weighting.
  - Name-only fallback path for the ~3.3% of Source 2/3 entities with missing or empty addresses.
- **Asymmetric Frequency Capping & Selective Expansion**:
  - Address tokens capped at 5,000 to eliminate street number and generic locality posting-list bloat.
  - Name tokens capped at 10,000 (determined via empirical comparison against 15,000 and 20,000; cap=10k achieved 110.1 aggregate q/s with 92.38% Recall@200, whereas higher caps collapsed throughput by 30–40% without increasing top-200 recall due to candidate list dilution).
  - Selective posting list expansion skips traversing tokens $>5,000$ frequency whenever a query record possesses at least one distinctive token ($\le 5,000$ frequency).
- **Candidate Set Size & Recall Ceiling (Full Dataset Evaluation)**:
  - Final candidate budget fixed at $K=200$ per Source 1 entity (mean: 199.42, median: 200.0, reduction ratio: **99.9981%** across all 2,206,821 S1 entities × 10,320,219 S2+S3 reference records).
  - Evaluated on all **7,638,365 true ground-truth pairs**:
    - **Recall@50**: 88.79% (6,781,778 true pairs)
    - **Recall@100**: 90.95% (6,946,976 true pairs)
    - **Recall@150**: 91.73% (7,006,359 true pairs)
    - **Recall@200**: **92.20%** (7,042,261 true pairs)
  - Full-scale pipeline throughput: **102.1 queries/sec aggregate** across the entire 2.206M query dataset, generating **440,091,505 candidate pairs** in `output/candidate_pairs.tsv` with zero data loss via per-partition checkpointing (`checkpoints/candidates_india.tsv` and `checkpoints/candidates_us.tsv`).
  - Under the competition's precision-weighted $F_{0.5}$ metric ($\beta=0.5$), this 92.20% recall ceiling paired with a 99.9981% reduction ratio provides the optimal candidate pool for high-precision downstream feature extraction and classification.

---

## 4. Matching Model

### Features used:
- **Name features:**
  - Token Jaccard similarity and containment ratio
  - Levenshtein distance and Normalized Levenshtein similarity
  - RapidFuzz token sort ratio and token set ratio
  - Exact name match boolean flag
  - Prefix and suffix match indicators
- **Address features:**
  - Token overlap Jaccard similarity
  - Numeric address token and house number exact match
  - Address bigram overlap similarity
  - Address length difference and missing address flag
- **Cross-field & Meta features:**
  - Country consistency indicator
  - Name-to-address length ratio

### Model configuration:
- **Model type:** LightGBM (Pairwise Binary Classification / Gradient Boosted Trees)
- **Threshold selection method:** Grid sweep optimizing macro $F_{0.5}$ on out-of-fold validation pairs, incorporating explicit singleton evaluation (empty match predictions evaluated at 1.0 for true singletons).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [TODO: fill in once Milestone 7-10 numbers are final]
- **Common false positives (wrong merges):** [TODO: fill in once Milestone 7-10 numbers are final]
- **Common false negatives (missed matches):** [TODO: fill in once Milestone 7-10 numbers are final]

---

## 6. Conclusion

Our solution achieves high-recall candidate generation combined with precision-heavy LightGBM classification tailored to the challenge's $F_{0.5}$ metric. By addressing legal suffix canonicalization, cross-script compatibility, and singleton dynamics, the system resolves business entities accurately and efficiently at scale.

---

## Appendix

### A. Code Artefacts
The complete runnable codebase is organized under `code/business_entity_resolution/`:
- `src/business_entity_resolution/normalize.py`: Unicode-aware text canonicalization.
- `src/business_entity_resolution/blocking.py`: Inverted index candidate generation and IDF ranking.
- `src/business_entity_resolution/features.py`: Multi-field lexical and fuzzy feature extraction.
- `src/business_entity_resolution/model.py`: LightGBM pairwise classifier training and inference.
- `src/business_entity_resolution/scoring.py`: Exact competition macro $F_{0.5}$ metric evaluator.
- `src/business_entity_resolution/io_utils.py`: TSV format validation and output writers.

**Entry Points:**
- `python code/business_entity_resolution/scripts/run_train.py --stage all`: Executes training and full pipeline inference.
- `bash scripts/build_submission.sh`: Validates output formats and builds `<TEAM_NAME>_submission.zip`.

### B. Additional Results
[TODO: fill in once Milestone 7-10 numbers are final]
