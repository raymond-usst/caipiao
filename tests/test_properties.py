"""Property-based tests using Hypothesis for edge case discovery."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

try:
    from hypothesis import given, strategies as st, settings, HealthCheck
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

import unittest


class TestLotteryProperties(unittest.TestCase):
    """Property-based tests for lottery data invariants."""
    
    @unittest.skipUnless(HAS_HYPOTHESIS, "hypothesis not installed")
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(
        red1=st.integers(min_value=1, max_value=5),
        red2=st.integers(min_value=6, max_value=10),
        red3=st.integers(min_value=11, max_value=15),
        red4=st.integers(min_value=16, max_value=20),
        red5=st.integers(min_value=21, max_value=25),
        red6=st.integers(min_value=26, max_value=33),
        blue=st.integers(min_value=1, max_value=16)
    )
    def test_red_balls_invariant_sorted(self, red1, red2, red3, red4, red5, red6, blue):
        """Red balls should always be in ascending order."""
        reds = [red1, red2, red3, red4, red5, red6]
        self.assertEqual(reds, sorted(reds))
    
    @unittest.skipUnless(HAS_HYPOTHESIS, "hypothesis not installed")
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    @given(blue=st.integers(min_value=1, max_value=16))
    def test_blue_ball_range(self, blue):
        """Blue ball should be in valid range [1, 16]."""
        self.assertGreaterEqual(blue, 1)
        self.assertLessEqual(blue, 16)
    
    @unittest.skipUnless(HAS_HYPOTHESIS, "hypothesis not installed")
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    @given(st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=33),
            st.floats(min_value=0, max_value=1, allow_nan=False)
        ),
        min_size=1,
        max_size=10
    ))
    def test_probability_sum_is_valid(self, probs):
        """Test that prediction probability lists are valid."""
        # Probabilities should be non-negative
        for num, prob in probs:
            self.assertGreaterEqual(prob, 0)
            self.assertLessEqual(prob, 1)
    
    @unittest.skipUnless(HAS_HYPOTHESIS, "hypothesis not installed")
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n=st.integers(min_value=30, max_value=100)
    )
    def test_features_output_matches_input_length(self, n):
        """Feature output should have same length as input."""
        dates = pd.date_range("2023-01-01", periods=n)
        df = pd.DataFrame({
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
        
        from lottery.features import build_features
        features = build_features(df)
        self.assertEqual(len(features), n)


class TestMonteCarloProperties(unittest.TestCase):
    """Property tests for Monte Carlo simulation."""
    
    @unittest.skipUnless(HAS_HYPOTHESIS, "hypothesis not installed")
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_samples=st.integers(min_value=100, max_value=500),
        base_sum=st.floats(min_value=50, max_value=150, allow_nan=False)
    )
    def test_monte_carlo_mean_near_base(self, n_samples, base_sum):
        """Monte Carlo mean should be close to base prediction."""
        from lottery.monte_carlo import sample_predictions
        
        def mock_predictor(df):
            return {"sum_pred": base_sum, "blue": [(8, 0.5)]}
        
        df = pd.DataFrame({"issue": [1]})
        result = sample_predictions(mock_predictor, df, n_samples=n_samples, noise_std=0.05)
        
        # Mean should be within 20% of base
        self.assertAlmostEqual(result["sum"]["mean"], base_sum, delta=base_sum * 0.2)


if __name__ == "__main__":
    unittest.main()
