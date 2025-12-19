"""
Reproducibility Script for TGAlign Paper
========================================
This script reproduces the 5-fold cross-validation benchmarks presented in:
"High-Accuracy, Ultrafast DNA Barcode Identification via Statistical Sketching"

Prerequisites:
1. Install tgalign: `pip install .` (from the repo root)
2. Run from repo root: `python benchmarks/reproduce_paper.py`

Data Source:
All datasets are permanently archived on Zenodo (DOI: 10.5281/zenodo.17973054)
to ensure exact reproducibility of the results.
"""

import gzip
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import requests
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# =============================================================================
# LIBRARY IMPORTS & SETUP
# =============================================================================

try:
    from tgalign import TGAlignIndex
    print("SUCCESS: TGAlign library loaded.")
except ImportError:
    print("CRITICAL ERROR: 'tgalign' package not found.")
    print("Please install it first using: pip install .")
    sys.exit(1)

# USEARCH configuration
USEARCH_BINARY = "usearch12"
USEARCH_URL = "https://github.com/rcedgar/usearch12/releases/download/v12.0-beta1/usearch_linux_x86_12.0-beta"

def ensure_usearch():
    """Checks for USEARCH binary in PATH or CWD, downloads if missing."""
    if shutil.which(USEARCH_BINARY):
        return os.path.abspath(shutil.which(USEARCH_BINARY))
    
    if os.path.exists(USEARCH_BINARY):
        return os.path.abspath(USEARCH_BINARY)

    print(f"\n--- USEARCH binary not found. Downloading {USEARCH_BINARY}... ---")
    try:
        subprocess.run(f"wget -q {USEARCH_URL} -O {USEARCH_BINARY}", shell=True, check=True)
        subprocess.run(f"chmod +x {USEARCH_BINARY}", shell=True, check=True)
        print("--- USEARCH downloaded successfully. ---\n")
        return os.path.abspath(USEARCH_BINARY)
    except subprocess.CalledProcessError:
        print("ERROR: Failed to download USEARCH. Please install it manually.")
        sys.exit(1)

USEARCH_EXE = ensure_usearch()

def run_command(command: str):
    subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL)

# =============================================================================
# DATA HELPERS
# =============================================================================

@dataclass
class BenchmarkCase:
    name: str
    min_len: int
    max_len: int
    archive_path: str
    tax_parser: Callable[[str], Tuple[str, str]]
    url: str  # Mandatory now (Zenodo)
    prep_method: str = 'filter'
    fasta_path_in_archive: Optional[str] = None
    taxonomy_path_in_archive: Optional[str] = None
    fwd_primer: Optional[str] = None
    rev_primer: Optional[str] = None
    usearch_id: float = 0.97
    is_balanced_10: bool = False

def iupac_to_regex(iupac_string: str) -> str:
    iupac_map = {
        'A': 'A', 'C': 'C', 'G': 'G', 'T': 'T',
        'R': '[AG]', 'Y': '[CT]', 'S': '[GC]', 'W': '[AT]',
        'K': '[GT]', 'M': '[AC]', 'B': '[CGT]', 'D': '[AGT]',
        'H': '[ACT]', 'V': '[ACG]', 'N': '[ATGC]'
    }
    return "".join(iupac_map.get(base, 'N') for base in iupac_string)

def parse_ncbi_taxonomy(header: str) -> Tuple[str, str]:
    parts = header.split()
    for i, part in enumerate(parts):
        if i + 1 < len(parts) and part[0].isupper() and len(part) > 2 and parts[i+1][0].islower():
            genus = part
            species = f"{part} {parts[i+1]}"
            if i + 2 < len(parts) and parts[i+2][0].islower():
                species += f" {parts[i+2]}"
            return species, genus
    return "Unknown", "Unknown"

def parse_greengenes_taxonomy(header: str) -> Tuple[str, str]:
    matches = re.findall(r'([kpcofgs])__([^;]*)', header)
    tax_map = {match[0]: match[1] for match in matches}
    genus = tax_map.get('g', '').strip()
    species_part = tax_map.get('s', '').strip()
    if species_part:
        label = f"{genus} {species_part}" if genus else species_part
    elif genus:
        label = genus
    elif tax_map.get('f', '').strip():
        label = tax_map.get('f').strip()
    else:
        label = "Unknown"
    return label, (genus if genus else "Unknown")

def read_fasta(file_path: str) -> Dict[str, str]:
    sequences = {}
    _open = gzip.open if file_path.endswith((".gz", ".gzip")) else open
    with _open(file_path, 'rt', errors='ignore') as f:
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

def insilico_pcr(sequences: Dict[str, str], fwd_primer: str, rev_primer: str) -> Dict[str, str]:
    print("    -> Performing in-silico PCR...")
    fragments = {}
    fwd_regex = re.compile(iupac_to_regex(fwd_primer))
    rev_rc = rev_primer.translate(str.maketrans("ATGC", "TACG"))[::-1]
    rev_regex_rc = re.compile(iupac_to_regex(rev_rc))

    for seq_id, seq in sequences.items():
        fwd_match = fwd_regex.search(seq)
        if fwd_match:
            rev_match = rev_regex_rc.search(seq, fwd_match.start())
            if rev_match:
                fragments[seq_id] = seq[fwd_match.start():rev_match.end()]
    print(f"    -> Extracted {len(fragments)} amplicons.")
    return fragments

def get_dataset_arrays(case: BenchmarkCase) -> Tuple[np.ndarray, np.ndarray]:
    print(f"\n--- [Data Prep] {case.name} ---")
    
    os.makedirs("data_cache", exist_ok=True)
    local_archive = os.path.join("data_cache", case.archive_path)

    # 1. Acquire Data (Download from Zenodo if missing)
    if not os.path.exists(local_archive):
        print(f"    -> Downloading frozen dataset from Zenodo...")
        run_command(f"wget -q --no-check-certificate '{case.url}' -O {local_archive}")

    # 2. Extract/Select File
    fasta_to_read = local_archive
    if local_archive.endswith((".tar.gz", ".tgz")):
        # Extract if needed
        # Check if already extracted
        if case.fasta_path_in_archive:
            extracted_path = os.path.join("data_cache", case.fasta_path_in_archive)
            if not os.path.exists(extracted_path):
                print("    -> Extracting archive...")
                run_command(f"tar -xzf {local_archive} -C data_cache")
            fasta_to_read = extracted_path

    # 3. Read and Parse
    full_db = read_fasta(fasta_to_read)
    
    tax_map = {}
    if case.taxonomy_path_in_archive:
        tax_path = os.path.join("data_cache", case.taxonomy_path_in_archive)
        if os.path.exists(tax_path):
            with open(tax_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2: tax_map[parts[0]] = parts[1]

    raw_sequences, raw_labels = [], []
    for header, seq in full_db.items():
        if 'N' in seq: continue
        tax_source = tax_map.get(header.split()[0]) if tax_map else header
        if tax_source:
            species, _ = case.tax_parser(tax_source)
            if species != "Unknown":
                raw_sequences.append(seq)
                raw_labels.append(species)

    # 4. PCR or Filtering
    if case.prep_method == 'pcr':
        pcr_input = {i: seq for i, seq in enumerate(raw_sequences)}
        pcr_results = insilico_pcr(pcr_input, case.fwd_primer, case.rev_primer)
        sequences, labels = [], []
        for idx, amplicon in pcr_results.items():
            if case.min_len <= len(amplicon) <= case.max_len:
                sequences.append(amplicon)
                labels.append(raw_labels[idx])
    else:
        sequences, labels = [], []
        for s, l in zip(raw_sequences, raw_labels):
            if case.min_len <= len(s) <= case.max_len:
                sequences.append(s)
                labels.append(l)

    # 5. Balancing
    if case.is_balanced_10:
        print("    -> Balancing dataset (max 10 per genus)...")
        sequences_np = np.array(sequences, dtype=object)
        labels_np = np.array(labels, dtype=object)
        indices_to_keep = []
        genus_counts = defaultdict(int)
        for i, l in enumerate(labels):
            try:
                _, genus = case.tax_parser(l)
                if genus != "Unknown" and genus_counts[genus] < 10:
                    indices_to_keep.append(i)
                    genus_counts[genus] += 1
            except Exception: pass
        sequences = sequences_np[indices_to_keep].tolist()
        labels = labels_np[indices_to_keep].tolist()

    print(f"    -> Final count: {len(sequences)} sequences.")
    return np.array(sequences), np.array(labels)

# =============================================================================
# WRAPPERS FOR MODELS
# =============================================================================

class UsearchWrapper:
    def __init__(self, id_threshold: float, expert_mode: bool = False):
        self.name = f"USEARCH ({'expert' if expert_mode else 'global'}, id={id_threshold})"
        self.id_threshold = id_threshold
        self.expert_mode = expert_mode
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "db.fasta")

    def build(self, train_db: Dict[str, str]):
        with open(self.db_path, "w") as f:
            for seq_id, seq in train_db.items():
                f.write(f">{seq_id}\n{seq}\n")

    def search(self, query_sequences: List[str]) -> List[str]:
        query_path = os.path.join(self.temp_dir, "q.fasta")
        output_path = os.path.join(self.temp_dir, "hits.txt")
        
        with open(query_path, "w") as f:
            for i, seq in enumerate(query_sequences):
                f.write(f">q{i}\n{seq}\n")

        cmd = [
            USEARCH_EXE, '-usearch_global', query_path,
            '-db', self.db_path,
            '-id', str(self.id_threshold),
            '-blast6out', output_path,
            '-strand', 'both', 
            '-maxaccepts', '1', '-maxrejects', '32', '-threads', '1'
        ]
        if self.expert_mode:
            cmd.extend(['-query_cov', '0.9'])

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        hits = {f"q{i}": "Unknown" for i in range(len(query_sequences))}
        if os.path.exists(output_path):
            with open(output_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        hits[parts[0]] = parts[1].rsplit('_', 1)[0]
        return [hits[f"q{i}"] for i in range(len(query_sequences))]

    def cleanup(self):
        shutil.rmtree(self.temp_dir)

class TGAlignWrapper:
    def __init__(self, name="TGAlign"):
        self.name = name
        # The TGA logic (tiling for fragments) is now AUTOMATIC inside the model
        self.model = TGAlignIndex(k=11, s=9, dim=4096, distance_threshold=0.8)

    def build(self, train_db: Dict[str, str]):
        self.model.build(train_db)

    def search(self, queries: List[str]) -> List[str]:
        return self.model.search(queries)
    
    def cleanup(self):
        pass

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def print_summary(results: Dict[str, Dict[str, List[float]]], title: str):
    print(f"\n=== RESULTS: {title} ===")
    models = sorted(results.keys())
    metrics = ["Accuracy", "Macro F1", "Time(Build, s)", "Time(Query, ms)"]
    
    # Print Header
    print(f"{'Metric':<20} | " + " | ".join([f"{m:<25}" for m in models]))
    print("-" * 80)

    for metric in metrics:
        row = f"{metric:<20}"
        for m in models:
            vals = results[m][metric]
            row += f" | {np.mean(vals):.4f} +/- {np.std(vals):.4f}"
        print(row)

    # T-Test
    tg = next((m for m in models if "TGAlign" in m), None)
    us = next((m for m in models if "USEARCH" in m), None)
    if tg and us:
        p = ttest_rel(results[tg]["Accuracy"], results[us]["Accuracy"]).pvalue
        print(f"\nPaired T-Test ({tg} vs {us}): p = {p:.5f}")

def main():
    random.seed(42)
    np.random.seed(42)

    # Define the 5 Benchmarks (Pointing to Zenodo)
    cases = [
        BenchmarkCase(name="16S V4 (Greengenes)", prep_method='pcr', min_len=200, max_len=300,
                  archive_path="gg_16s_greengenes.tar.gz",
                  url="https://zenodo.org/records/17973054/files/gg_16s_greengenes.tar.gz?download=1",
                  fasta_path_in_archive="gg_13_8_otus/rep_set/99_otus.fasta",
                  taxonomy_path_in_archive="gg_13_8_otus/taxonomy/99_otu_taxonomy.txt",
                  fwd_primer="GTGCCAGCMGCCGCGGTAA", rev_primer="GGACTACHVGGGTWTCTAAT",
                  usearch_id=0.97, tax_parser=parse_greengenes_taxonomy),

        BenchmarkCase(name="ITS (Fungi)", prep_method='filter', min_len=200, max_len=1000,
                  archive_path="ncbi_its_fungi.fasta", 
                  url="https://zenodo.org/records/17973054/files/ncbi_its_fungi.fasta?download=1",
                  usearch_id=0.985, tax_parser=parse_ncbi_taxonomy),

        BenchmarkCase(name="COI Full-Length", min_len=600, max_len=700,
            archive_path="ncbi_coi.fasta", 
            url="https://zenodo.org/records/17973054/files/ncbi_coi.fasta?download=1",
            usearch_id=0.97, tax_parser=parse_ncbi_taxonomy),
        
        # Uses same file as Full-Length, just cached locally
        BenchmarkCase(name="COI Fragments", min_len=600, max_len=700,
            archive_path="ncbi_coi.fasta", 
            url="https://zenodo.org/records/17973054/files/ncbi_coi.fasta?download=1",
            usearch_id=0.97, tax_parser=parse_ncbi_taxonomy),

        BenchmarkCase(name="COI Balanced", min_len=600, max_len=700,
            archive_path="ncbi_coi.fasta", 
            url="https://zenodo.org/records/17973054/files/ncbi_coi.fasta?download=1",
            usearch_id=0.97, tax_parser=parse_ncbi_taxonomy,
            is_balanced_10=True)
    ]

    for case in cases:
        sequences, labels = get_dataset_arrays(case)
        if len(sequences) < 50: 
            print("Skipping due to insufficient data."); continue

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = defaultdict(lambda: defaultdict(list))
        
        is_fragment_test = "Fragments" in case.name
        is_balanced = "Balanced" in case.name

        for fold, (train_idx, test_idx) in enumerate(skf.split(sequences, labels)):
            print(f"  Running Fold {fold+1}/5...")
            
            # Prepare Data
            train_db = {f"{l.replace(' ','_')}_{i}": s for i, (s, l) in enumerate(zip(sequences[train_idx], labels[train_idx]))}
            
            # Prepare Queries (Simulate fragments if needed)
            queries, ground_truth = [], []
            for seq, lbl in zip(sequences[test_idx], labels[test_idx]):
                lbl = lbl.replace(' ', '_')
                if is_fragment_test and len(seq) > 300:
                    start = random.randint(0, len(seq) - 300)
                    queries.append(seq[start : start + 300])
                    ground_truth.append(lbl)
                else:
                    queries.append(seq)
                    ground_truth.append(lbl)

            # Define Competitors
            # TGAlign is always parameter-free (auto-tiling)
            tg_wrapper = TGAlignWrapper()
            
            if is_fragment_test:
                # For Fragments, we compare against BOTH Expert and Standard
                comps = [
                    tg_wrapper,
                    UsearchWrapper(case.usearch_id, expert_mode=True),  # The "Expert"
                    UsearchWrapper(case.usearch_id, expert_mode=False)  # The "Naive"
                ]
            else:
                # For standard datasets, we just use standard global alignment
                comps = [
                    tg_wrapper,
                    UsearchWrapper(case.usearch_id, expert_mode=False)
                ]

            # Run Benchmarks
            for model in comps:
                # Build
                t0 = time.time()
                model.build(train_db)
                t_build = time.time() - t0
                
                # Query
                t0 = time.time()
                preds = model.search(queries)
                t_query = time.time() - t0
                
                # Metrics
                acc = accuracy_score(ground_truth, preds)
                f1 = f1_score(ground_truth, preds, average='macro', zero_division=0)
                
                results[model.name]["Accuracy"].append(acc)
                results[model.name]["Macro F1"].append(f1)
                results[model.name]["Time(Build, s)"].append(t_build)
                results[model.name]["Time(Query, ms)"].append((t_query * 1000) / len(queries))
                
                model.cleanup()

        print_summary(results, case.name)

if __name__ == "__main__":
    main()
