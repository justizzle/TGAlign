"""
Ablation Study: Validating the TGA Architectural Heuristic
==========================================================
This script reproduces the ablation results (Table S1) by forcing
TGAlign to disable its automatic windowing/tiling logic.

This proves that the "Geometry Alignment" (Tiling) is the causal driver 
of performance on fragments, not just the Syncmer sketching.

Usage: python benchmarks/run_ablation.py
"""

import os
import random
import sys
import time
import numpy as np
import requests
import tarfile
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

try:
    from tgalign import TGAlignIndex
    # We also import the internal C++ backend to bypass the auto-logic if needed,
    # or we simply subclass TGAlignIndex to override .build()
except ImportError:
    print("Error: tgalign not installed. Run 'pip install .'")
    sys.exit(1)

# =============================================================================
# ABLATED MODEL (The "Naive" Version)
# =============================================================================

class NaiveTGAlignIndex(TGAlignIndex):
    """
    A crippled version of TGAlign that disables Task-Geometry Alignment (Tiling).
    It treats every reference sequence as a single vector, regardless of length.
    """
    def build(self, reference_db):
        print("  [Ablation] Building index WITHOUT tiling logic...")
        # Standard FAISS setup
        import faiss
        nlist = max(1, min(len(reference_db) * 5 // 40, 200))
        quantizer = faiss.IndexFlatL2(self.vector_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist)

        sequences_to_sketch = []
        current_idx = 0

        for ref_id, seq in reference_db.items():
            # ABLATION: Always sketch the full sequence as one unit.
            # No windowing check. No tiling loop.
            sequences_to_sketch.append(seq)
            self.label_map[current_idx] = ref_id
            current_idx += 1

        if not sequences_to_sketch: return

        # Call the C++ backend
        sketches = self._sketch_batch(sequences_to_sketch)
        self.index.train(sketches)
        self.index.add(sketches)

# =============================================================================
# DATA PREP
# =============================================================================

def parse_ncbi_taxonomy(header: str) -> str:
    """Strict Species-Level Parser (Same as Main Benchmark)"""
    parts = header.split()
    for i, part in enumerate(parts):
        # Look for 'Genus species' pattern
        if i + 1 < len(parts) and part[0].isupper() and len(part) > 2 and parts[i+1][0].islower():
            genus = part
            species = f"{part} {parts[i+1]}"
            # Optional: Add subspecies if present
            if i + 2 < len(parts) and parts[i+2][0].islower():
                species += f" {parts[i+2]}"
            return species
    return "Unknown"

def get_coi_data():
    """Downloads and prepares the COI dataset from Zenodo."""
    url = "https://zenodo.org/records/17973054/files/ncbi_coi.fasta?download=1"
    os.makedirs("data_cache", exist_ok=True)
    path = "data_cache/ncbi_coi.fasta"
    
    if not os.path.exists(path):
        print("Downloading COI data...")
        import subprocess
        subprocess.run(f"wget -q '{url}' -O {path}", shell=True)
    
    # Read FASTA
    raw_seqs, raw_labels = [], []
    with open(path, 'r') as f:
        header = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and "Unknown" not in header:
                    # USE STRICT PARSER
                    label = parse_ncbi_taxonomy(header)
                    if label != "Unknown":
                        raw_seqs.append("".join(seq))
                        raw_labels.append(label)
                header = line[1:]
                seq = []
            else:
                seq.append(line.upper())
        # Last seq
        if header:
             label = parse_ncbi_taxonomy(header)
             if label != "Unknown":
                 raw_seqs.append("".join(seq))
                 raw_labels.append(label)

    # FILTER RARE CLASSES
    from collections import Counter
    counts = Counter(raw_labels)
    valid_indices = [i for i, l in enumerate(raw_labels) if counts[l] >= 5]
    
    print(f"Original: {len(raw_labels)}. Filtered: {len(valid_indices)} sequences.")
    return np.array(raw_seqs)[valid_indices], np.array(raw_labels)[valid_indices]
# =============================================================================
# RUN ABLATION BENCHMARK
# =============================================================================

def run_ablation():
    random.seed(42)
    np.random.seed(42)
    
    print("\n=== RUNNING ABLATION STUDY (Table S1) ===")
    
    sequences, labels = get_coi_data()
    print(f"Loaded {len(sequences)} COI sequences.")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        "Standard (TGA)": {"acc": [], "f1": []},
        "Ablated (Naive)": {"acc": [], "f1": []}
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\nFold {fold+1}/5")
        
        train_db = {f"{l}_{i}": s for i, (s, l) in enumerate(zip(sequences[train_idx], labels[train_idx]))}
        
        # FRAGMENT GENERATION (Critical for this test)
        queries, ground_truth = [], []
        for seq, lbl in zip(sequences[test_idx], labels[test_idx]):
            if len(seq) > 300:
                start = random.randint(0, len(seq) - 300)
                queries.append(seq[start : start + 300])
                ground_truth.append(lbl)
            else:
                queries.append(seq)
                ground_truth.append(lbl)

        # 1. Standard Model (TGA Active)
        model_std = TGAlignIndex() # Uses auto-tiling
        model_std.build(train_db)
        preds_std = model_std.search(queries)
        
        acc_std = accuracy_score(ground_truth, preds_std)
        results["Standard (TGA)"]["acc"].append(acc_std)
        print(f"  Standard (TGA): {acc_std:.4f}")

        # 2. Ablated Model (Naive)
        model_abl = NaiveTGAlignIndex() # Uses subclass override
        model_abl.build(train_db)
        preds_abl = model_abl.search(queries)
        
        acc_abl = accuracy_score(ground_truth, preds_abl)
        results["Ablated (Naive)"]["acc"].append(acc_abl)
        print(f"  Ablated (Naive): {acc_abl:.4f}")

    print("\n=== FINAL ABLATION RESULTS ===")
    print(f"Standard (TGA) Accuracy: {np.mean(results['Standard (TGA)']['acc']):.4f}")
    print(f"Ablated (Naive) Accuracy: {np.mean(results['Ablated (Naive)']['acc']):.4f}")
    
    drop = np.mean(results['Standard (TGA)']['acc']) - np.mean(results['Ablated (Naive)']['acc'])
    print(f"Performance Collapse due to Ablation: -{drop*100:.2f}%")

if __name__ == "__main__":
    run_ablation()
