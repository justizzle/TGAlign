#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace py = pybind11;

inline int nt_to_int_sync(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return -1;
    }
}

uint64_t string_to_canonical_kmer(const std::string& s, int k) {
    uint64_t forward = 0, reverse = 0;
    for (int i = 0; i < k; ++i) {
        int nt = nt_to_int_sync(s[i]);
        if (nt == -1) return -1;
        forward = (forward << 2) | nt;
        reverse = (reverse >> 2) | ((uint64_t)(3 - nt) << (2 * (k - 1)));
    }
    return std::min(forward, reverse);
}

uint64_t hash64(uint64_t key) {
    key = (~key) + (key << 21); key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}

void create_syncmer_sketch_single(const std::string& sequence, int k, int s, int vector_dim, float* out_sketch) {
    std::fill(out_sketch, out_sketch + vector_dim, 0.0f);
    if (sequence.length() < (unsigned int)k) return;

    for (size_t i = 0; i <= sequence.length() - k; ++i) {
        uint64_t min_smer_val = -1;
        int min_smer_pos = -1;

        for (int j = 0; j <= k - s; ++j) {
            uint64_t smer_val = string_to_canonical_kmer(sequence.substr(i + j, s), s);
            if (smer_val != (uint64_t)-1 && (min_smer_pos == -1 || smer_val < min_smer_val)) {
                min_smer_val = smer_val;
                min_smer_pos = j;
            }
        }

        if (min_smer_pos == 0) {
            uint64_t kmer_val = string_to_canonical_kmer(sequence.substr(i, k), k);
            if (kmer_val != (uint64_t)-1) {
                uint64_t hashed_kmer = hash64(kmer_val);
                out_sketch[hashed_kmer % vector_dim]++;
            }
        }
    }

    double norm_sq = 0.0;
    for (int i = 0; i < vector_dim; ++i) norm_sq += out_sketch[i] * out_sketch[i];
    if (norm_sq > 0) {
        float norm = std::sqrt(norm_sq);
        for (int i = 0; i < vector_dim; ++i) out_sketch[i] /= norm;
    }
}

py::array_t<float> create_syncmer_sketches_batch_cpp(const std::vector<std::string>& sequences, int k, int s, int vector_dim) {
    const size_t num_sequences = sequences.size();
    auto all_sketches_np = py::array_t<float>(std::vector<size_t>{num_sequences, (size_t)vector_dim});
    float* all_sketches_ptr = static_cast<float*>(all_sketches_np.request().ptr);

    for (size_t i = 0; i < num_sequences; ++i) {
        create_syncmer_sketch_single(sequences[i], k, s, vector_dim, all_sketches_ptr + i * vector_dim);
    }
    return all_sketches_np;
}

PYBIND11_MODULE(_tgalign_cpp, m) {
    m.doc() = "High-performance C++ backend for TGAlign";
    m.def("create_syncmer_sketches_batch", &create_syncmer_sketches_batch_cpp, "Creates syncmer sketches");
}
