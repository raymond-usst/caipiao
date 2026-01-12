"""Unit tests for lottery/analyzer.py"""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from lottery import analyzer

class TestAnalyzer(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create mock data."""
        n = 50
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
    
    def test_frequency_stats_returns_dict(self):
        """frequency_stats should return a dict with 'red' and 'blue'."""
        result = analyzer.frequency_stats(self.df)
        self.assertIsInstance(result, dict)
        self.assertIn("red", result)
        self.assertIn("blue", result)
        self.assertEqual(len(result["red"]), 33)  # 1-33
        self.assertEqual(len(result["blue"]), 16)  # 1-16
        
    def test_hot_cold_returns_lists(self):
        """hot_cold should return two lists."""
        probs = {i: 1.0/33 for i in range(1, 34)}
        probs[1] = 0.5  # Make 1 hot
        probs[33] = 0.001  # Make 33 cold
        hot, cold = analyzer.hot_cold(probs, top_k=3)
        self.assertIsInstance(hot, list)
        self.assertIsInstance(cold, list)
        self.assertEqual(len(hot), 3)
        self.assertEqual(len(cold), 3)
        self.assertIn(1, hot)
        self.assertIn(33, cold)
        
    def test_calculate_omission_shape(self):
        """calculate_omission should return DataFrame with correct shape."""
        result = analyzer.calculate_omission(self.df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape[0], len(self.df))
        self.assertEqual(result.shape[1], 33)
        
    def test_basic_stats_returns_dict(self):
        """basic_stats should return a dict with expected keys."""
        result = analyzer.basic_stats(self.df)
        self.assertIsInstance(result, dict)
        self.assertIn("red_mean", result)
        self.assertIn("sum_mean", result)

if __name__ == "__main__":
    unittest.main()
