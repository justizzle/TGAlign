import unittest
import os
import shutil
import tempfile
from tgalign import TGAlignIndex

class TestTGAlign(unittest.TestCase):
    def setUp(self):
        # Create a dummy test case
        self.ref_db = {
            "ref1_full": "ATGC" * 100,  # 400bp (Should be tiled)
            "ref2_short": "CGTA" * 50,  # 200bp (Should NOT be tiled)
        }
        self.queries = [
            "ATGC" * 50,   # Matches ref1 (fragment)
            "CGTA" * 50,   # Matches ref2 (exact)
            "AAAA" * 50    # Unknown
        ]
        
    def test_end_to_end(self):
        print("\n--- Running Functional Test ---")
        
        # 1. Initialize
        # Using small dim for speed, standard k/s
        idx = TGAlignIndex(k=11, s=9, dim=512, distance_threshold=0.5)
        
        # 2. Build
        print(f"Building index with {len(self.ref_db)} sequences...")
        idx.build(self.ref_db)
        
        # 3. Search
        print(f"Searching {len(self.queries)} queries...")
        results = idx.search(self.queries)
        
        print(f"Results: {results}")
        
        # 4. Assertions
        # The model strips suffixes after the last underscore (common for taxonomy IDs)
        # 'ref1_full' -> 'ref1'
        self.assertEqual("ref1", results[0]) 
        self.assertEqual("ref2", results[1])
        self.assertEqual("Unknown", results[2])
        
        print("✅ Test Passed: TGA correctly identified fragment and full match.")

if __name__ == '__main__':
    unittest.main()
