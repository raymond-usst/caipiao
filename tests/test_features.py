"""Unit tests for lottery/features.py"""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from lottery import features

class TestFeatures(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create mock data."""
        n = 100
        dates = pd.date_range("2023-01-01", periods=n)
        cls.df = pd.DataFrame({
            "issue": range(1, n+1),
            "draw_date": dates,
            "red1": np.random.randint(1, 6, n),
            "red2": np.random.randint(6, 11, n),
            "red3": np.random.randint(11, 16, n),
            "red4": np.random.randint(16, 21, n),
            "red5": np.random.randint(21, 26, n),
            "red6": np.random.randint(26, 34, n),
            "blue": np.random.randint(1, 17, n)
        })
    
    def test_build_features_returns_dataframe(self):
        """build_features should return a DataFrame."""
        result = features.build_features(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        
    def test_build_features_has_expected_columns(self):
        """build_features should produce expected columns."""
        result = features.build_features(self.df)
        expected_cols = ["ac", "span", "sum_tail", "odd_ratio", "big_ratio", 
                         "prime_ratio", "omit_red_mean", "omit_red_max", 
                         "omit_blue", "weekday"]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")
            
    def test_build_features_row_count_matches(self):
        """build_features output should have same row count as input."""
        result = features.build_features(self.df)
        self.assertEqual(len(result), len(self.df))
        
    def test_ac_values_range(self):
        """AC values should be non-negative."""
        result = features.build_features(self.df)
        self.assertTrue((result["ac"] >= 0).all())
        
    def test_odd_ratio_range(self):
        """odd_ratio should be between 0 and 1."""
        result = features.build_features(self.df)
        self.assertTrue((result["odd_ratio"] >= 0).all())
        self.assertTrue((result["odd_ratio"] <= 1).all())

if __name__ == "__main__":
    unittest.main()
