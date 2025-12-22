# TGAlign: Task-Geometry Alignment for DNA Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17973054.svg)](https://doi.org/10.5281/zenodo.17973054)

**TGAlign** is a high-performance, alignment-free DNA sequence search engine designed to solve the speed-accuracy bottleneck in modern bioinformatics.

By leveraging **Syncmer-based vectorization** and **Approximate Nearest Neighbor (ANN)** search, TGAlign achieves **20x faster query speeds** than industry-standard alignment tools (USEARCH v12) while maintaining statistical parity on difficult fragment identification tasks.

### Key Features
*   **🚀 C++ Accelerated:** Core sketching algorithms are implemented in optimized C++ with Pybind11 bindings for maximum throughput.
*   **🧬 Indel Robustness:** Uses **Syncmers** (Edgar, 2021) instead of standard Minimizers, allowing for high accuracy even in indel-heavy markers (ITS, 16S).
*   **🧩 Parameter-Free Geometry:** Automatically detects sequence length context ("Task-Geometry Alignment") to decompose long references, solving the "Fragment Problem" inherent in vector search without user tuning.
*   **📉 Sub-Millisecond Latency:** Average query times < 0.2ms per sequence.

---

## 📊 Performance Benchmark

TGAlign was benchmarked against **USEARCH v12 (Global Alignment)** on 5 datasets using 5-fold stratified cross-validation.

| Dataset | Metric | USEARCH (Expert Params) | TGAlign (Ours) | Difference |
| :--- | :--- | :--- | :--- | :--- |
| **16S V4** | Accuracy | 75.16% | **84.49%** | **+9.3%** |
| **ITS (Fungi)** | Accuracy | 43.27% | **50.81%** | **+7.5%** |
| **COI Fragments** | Accuracy | 70.97% | **70.57%** | (Parity, p > 0.05) |
| **COI Full** | Speed | 4.68 ms/query | **0.19 ms/query** | **~24x Faster** |

> **Validation:** The experimental design and benchmarking protocols were vetted by **Robert Edgar** (developer of USEARCH/MUSCLE).

---

## 📦 Installation

### Prerequisites
*   Python 3.8+
*   C++17 compliant compiler (GCC/Clang)
*   `faiss-cpu` (or `faiss-gpu`)

### From Source
```bash
git clone https://github.com/yourusername/tgalign.git
cd tgalign
pip install .

💻 Usage
TGAlign is designed to be used as a Python library for high-throughput pipelines.

from tgalign import TGAlignIndex
from tgalign.utils import read_fasta

# 1. Initialize Index
# TGAlign automatically handles parameterization for fragments vs full-length
index = TGAlignIndex()

# 2. Build Index (Reference Database)
ref_data = read_fasta("reference_database.fasta")
print(f"Indexing {len(ref_data)} sequences...")
index.build(ref_data)

# 3. Search (Query Sequences)
queries = ["ATGCGTAGCTAGCTAGCT...", "CGTAGCTAGCTAGCTAGC..."]
results = index.search(queries)

print(results) 
# Output: ['Genus_species_A', 'Genus_species_B']

🔬 Algorithm: Task-Geometry Alignment (TGA)
Standard vector search tools (like MinHash) often fail when matching short query fragments to long reference genomes because the geometric representation of a fragment is fundamentally different from that of a full gene.
TGA solves this by structurally aligning the algorithm to the data:
Syncmer Sketching: We utilize Open Syncmers (k=11, s=9) to select conserved context-dependent k-mers, which are robust to insertions/deletions.
Geometric Tiling: The indexer automatically detects reference sequences significantly longer than the query window (350bp). It decomposes these references into overlapping vector tiles.
Vector Quantization: Sketches are projected into a 4096-dimensional space and indexed using an Inverted File System (IVF) via FAISS.
This allows a 300bp fragment to find its exact geometric match within a 1500bp reference gene without the heavy computational cost of Smith-Waterman alignment.

🧪 Reproducing the Paper
To verify the results presented in the manuscript:
Install the package: pip install .
Run the reproduction script:
Bash
python benchmarks/reproduce_paper.py
Note: The script will automatically download the required public datasets (Greengenes, NCBI snapshots) and—if not present—the USEARCH v12 binary for comparison purposes.

📚 Citation
If you use TGAlign in your research, please cite:
Boone, J. (2025). High-Accuracy, Ultrafast DNA Barcode Identification via Statistical Sketching and Approximate Nearest Neighbor Search. bioRxiv. DOI: [Insert DOI]

🙏 Acknowledgements
We thank Robert Edgar (developer of USEARCH) for critical guidance on the experimental design, specifically regarding the application of Syncmers and the parameterization of benchmarks.
