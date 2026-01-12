"""Integration tests for full train-all → blend → predict pipeline."""
import sys
from pathlib import Path
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from lottery import blender


class TestPipelineIntegration(unittest.TestCase):
    """Test full pipeline integration."""
    
    @classmethod
    def setUpClass(cls):
        """Create mock data and temporary directory."""
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
        cls.temp_dir = tempfile.mkdtemp()
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup temporary directory."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_mock_predictor_produces_valid_output(self):
        """Test that mock predictor produces expected format."""
        def mock_predictor(df):
            return {
                "blue": [(np.random.randint(1, 17), 0.5)],
                "sum_pred": 100.0,
                "red": {i: [(i+1, 0.8)] for i in range(1, 7)}
            }
        
        result = mock_predictor(self.df)
        self.assertIn("blue", result)
        self.assertIn("sum_pred", result)
        self.assertIn("red", result)
        self.assertIsInstance(result["blue"], list)
        self.assertIsInstance(result["sum_pred"], float)
    
    def test_blend_with_mock_predictors(self):
        """Test blending with mock predictors."""
        def mock_predictor_1(df):
            return {
                "blue": [(1, 0.5), (2, 0.3)],
                "sum_pred": 100.0,
                "red": {i: [(i+1, 0.8)] for i in range(1, 7)}
            }
        
        def mock_predictor_2(df):
            return {
                "blue": [(2, 0.6), (1, 0.3)],
                "sum_pred": 110.0,
                "red": {i: [(i+2, 0.7)] for i in range(1, 7)}
            }
        
        predictors = [mock_predictor_1, mock_predictor_2]
        
        # Test blue blending
        blue_result = blender.blend_blue_latest(self.df, predictors)
        self.assertIsInstance(blue_result, tuple)
        self.assertEqual(len(blue_result), 2)
        
        # Test sum blending
        sum_result = blender.blend_sum_latest(self.df, predictors)
        self.assertIsInstance(sum_result, float)
        
        # Test red blending
        red_result = blender.blend_red_latest(self.df, predictors)
        self.assertIsInstance(red_result, dict)
    
    def test_features_build_in_pipeline(self):
        """Test feature building as part of pipeline."""
        from lottery.features import build_features
        
        features = build_features(self.df)
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.df))
        
        # Check essential columns exist
        expected_cols = ["ac", "span", "odd_ratio"]
        for col in expected_cols:
            self.assertIn(col, features.columns)


class TestDiversityMetrics(unittest.TestCase):
    """Test ensemble diversity metrics."""
    
    def test_pairwise_disagreement(self):
        """Test pairwise disagreement calculation."""
        from lottery.diversity import pairwise_disagreement
        
        # All same predictions = 0 disagreement
        preds_same = [{"blue": [(1, 0.5)]}, {"blue": [(1, 0.5)]}, {"blue": [(1, 0.5)]}]
        self.assertEqual(pairwise_disagreement(preds_same), 0.0)
        
        # All different = 1 disagreement
        preds_diff = [{"blue": [(1, 0.5)]}, {"blue": [(2, 0.5)]}, {"blue": [(3, 0.5)]}]
        self.assertEqual(pairwise_disagreement(preds_diff), 1.0)
    
    def test_correlation_diversity(self):
        """Test correlation diversity calculation."""
        from lottery.diversity import correlation_diversity
        
        n = 50
        df = pd.DataFrame({
            "issue": range(n),
            "draw_date": pd.date_range("2023-01-01", periods=n),
            "red1": np.random.randint(1, 6, n),
            "red2": np.random.randint(6, 11, n),
            "red3": np.random.randint(11, 16, n),
            "red4": np.random.randint(16, 21, n),
            "red5": np.random.randint(21, 26, n),
            "red6": np.random.randint(26, 34, n),
            "blue": np.random.randint(1, 17, n)
        })
        
        def pred1(df): return {"sum_pred": 100.0}
        def pred2(df): return {"sum_pred": 110.0}
        
        result = correlation_diversity([pred1, pred2], df, n_samples=10)
        self.assertIn("diversity_score", result)
        self.assertIn("n_models", result)


if __name__ == "__main__":
    unittest.main()
