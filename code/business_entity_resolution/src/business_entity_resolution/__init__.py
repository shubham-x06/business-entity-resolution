"""
business_entity_resolution  –  Entity resolution pipeline for the ML Challenge 2026.

Modules
-------
normalize : Text normalisation helpers (business names, addresses).
blocking  : Candidate-pair generation via MinHash LSH and rule-based blocking.
features  : Pairwise feature engineering (string similarity, token overlap, …).
model     : LightGBM wrapper for training and inference.
grouping  : Post-model grouping / transitive-closure logic.
pipeline  : End-to-end orchestration (train & infer entry-points).
io_utils  : I/O helpers (TSV reading, config loading, result writing).
"""

__version__ = "0.1.0"
