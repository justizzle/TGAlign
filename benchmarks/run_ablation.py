"""
Ablation Study: Validating the TGA Architectural Heuristic
"""

import os
import gzip
import random
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

try:
    from tgalign import TGAlignIndex
except ImportError:
    print("Error: tgalign not installed. Run 'pip install .'")
    sys.exit(1)

# =============================================================================
# 1. STRICT DATA LOADING (Copied from Original Script)
# =============================================================================

def parse_ncbi_taxonomy(header: str):
    """Strict parser matching the main benchmark."""
    parts = header.split()
    for i, part in enumerate(parts):
        # Look for Capitalized Genus + lowercase species
        if i + 1 < len(parts) and part[0].isupper() and len(part) > 2 and parts[i+1][0].islower():
            genus = part
            species = f"{part} {parts[i+1]}"
            # Optional: Add subspecies if present (matches original logic)
            if i + 2 < len(parts) and parts[i+2][0].islower():
                species += f" {parts[i+2]}"
            return species, genus
    return "Unknown", "Unknown"

def read_fasta(file_path):
    sequences = {}
    with open(file_path, 'r') as f:
        header = None
        current_seq = []
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if header: sequences["".join(current_seq)] = header
                header = line[1:]
                current_seq = []
            else:
                current_seq.append(line.upper().replace('U', 'T'))
        if header: sequences["".join(current_seq)] = header
    return {v: k for k, v in sequences.items() if k and v}

def get_coi_data_strict():
    """Downloads COI and applies strict N-filtering and taxonomy parsing."""
    url = "https://zenodo.org/records/17973054/files/ncbi_coi.fasta?download=1"
    os.makedirs("data_cache", exist_ok=True)
    path = "data_cache/ncbi_coi.fasta"
    
    # Download
    if not os.path.exists(path):
        print("Downloading COI data...")
        import subprocess
        subprocess.run(f"wget -q '{url}' -O {path}", shell=True)
    
    # Read raw
    full_db = read_fasta(path)
    print(f"Raw sequences in file: {len(full_db)}")

    sequences, labels = [], []
    
    # STRICT FILTERING LOOP
    for header, seq in full_db.items():
        # 1. Remove Ambiguous Bases (Critical for matching paper)
        if 'N' in seq: 
            continue

        # 2. Strict Taxonomy Parse
        species, _ = parse_ncbi_taxonomy(header)
        if species != "Unknown":
            sequences.append(seq)
            labels.append(species)

    # 3. Balancing (Optional, but ensures consistency if used)
    # The original fragment test wasn't balanced, but let's check counts
    print(f"Sequences after N-filtering and Parsing: {len(sequences)}")
    
    return np.array(sequences), np.array(labels)

# =============================================================================
# 2. ABLATED MODEL (Naive)
# =============================================================================

class NaiveTGAlignIndex(TGAlignIndex):
    """
    Overrides build() to disable Tiling.
    Treats every reference as a single vector.
    """
    def build(self, reference_db):
        import faiss
        # Standard Setup
        nlist = max(1, min(len(reference_db) * 5 // 40, 200))
        quantizer = faiss.IndexFlatL2(self.vector_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, nlist)

        sequences_to_sketch = []
        current_idx = 0

        for ref_id, seq in reference_db.items():
            # BYPASS TILING: Just add the whole sequence
            sequences_to_sketch.append(seq)
            self.label_map[current_idx] = ref_id
            current_idx += 1

        if not sequences_to_sketch: return

        # Sketch and Index
        sketches = self._sketch_batch(sequences_to_sketch)
        self.index.train(sketches)
        self.index.add(sketches)

# =============================================================================
# 3. RUN BENCHMARK
# =============================================================================

def run_ablation():
    random.seed(42)
    np.random.seed(42)
    
    print("\n=== RUNNING ABLATION STUDY (Strict Replication) ===")
    
    sequences, labels = get_coi_data_strict()
    
    # Filter for StratifiedKFold (min 5 per class)
    cnt = Counter(labels)
    valid_idx = [i for i, l in enumerate(labels) if cnt[l] >= 5]
    sequences = sequences[valid_idx]
    labels = labels[valid_idx]
    print(f"Final Dataset Size (min 5/class): {len(sequences)}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results_std = []
    results_abl = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\nFold {fold+1}/5")
        
        # Train DB
        train_db = {f"{l}_{i}": s for i, (s, l) in enumerate(zip(sequences[train_idx], labels[train_idx]))}
        
        # Test Fragments
        queries, ground_truth = [], []
        for seq, lbl in zip(sequences[test_idx], labels[test_idx]):
            if len(seq) > 300:
                start = random.randint(0, len(seq) - 300)
                queries.append(seq[start : start + 300])
                ground_truth.append(lbl)
            else:
                queries.append(seq)
                ground_truth.append(lbl)

        # 1. Standard (TGA Active)
        model_std = TGAlignIndex()
        model_std.build(train_db)
        acc_std = accuracy_score(ground_truth, model_std.search(queries))
        results_std.append(acc_std)
        print(f"  Standard (TGA): {acc_std:.4f}")

        # 2. Ablated (Naive)
        model_abl = NaiveTGAlignIndex()
        model_abl.build(train_db)
        acc_abl = accuracy_score(ground_truth, model_abl.search(queries))
        results_abl.append(acc_abl)
        print(f"  Ablated (Naive): {acc_abl:.4f}")

    # Summary
    mean_std = np.mean(results_std)
    mean_abl = np.mean(results_abl)
    print("\n=== FINAL RESULTS ===")
    print(f"Standard (TGA): {mean_std:.4f}")
    print(f"Ablated (Naive): {mean_abl:.4f}")
    print(f"Drop: {mean_abl - mean_std:.4f} ({(mean_abl - mean_std)*100:.2f}%)")

if __name__ == "__main__":
    run_ablation()
