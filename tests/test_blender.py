"""Unit tests for lottery/blender.py"""
import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from lottery import blender

class TestBlender(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Create mock data and predictors."""
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
        
        # Mock predictors
        def mock_predictor_1(df):
            return {
                "blue": [(1, 0.5), (2, 0.3), (3, 0.2)],
                "sum_pred": 100.0,
                "red": {i: [(i+1, 0.8)] for i in range(1, 7)}  # Positions 1-6
            }
        
        def mock_predictor_2(df):
            return {
                "blue": [(2, 0.6), (1, 0.3), (4, 0.1)],
                "sum_pred": 110.0,
                "red": {i: [(i+2, 0.7)] for i in range(1, 7)}  # Positions 1-6
            }
        
        cls.predictors = [mock_predictor_1, mock_predictor_2]
    
    def test_blend_blue_latest_returns_tuple(self):
        """blend_blue_latest should return (int, float)."""
        result = blender.blend_blue_latest(self.df, self.predictors)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], (int, np.integer))
        self.assertIsInstance(result[1], float)
        
    def test_blend_sum_latest_returns_float(self):
        """blend_sum_latest should return a float."""
        result = blender.blend_sum_latest(self.df, self.predictors)
        self.assertIsInstance(result, float)
        
    def test_blend_red_latest_returns_dict(self):
        """blend_red_latest should return a dict with 6 positions."""
        result = blender.blend_red_latest(self.df, self.predictors)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 6)

if __name__ == "__main__":
    unittest.main()
