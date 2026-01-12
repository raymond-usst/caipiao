
import unittest
import shutil
import tempfile
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lottery import database, nhits_model, tft_model, blender, pbt

class TestLotteryFast(unittest.TestCase):
    def setUp(self):
        # Create temp directory
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "test.db"
        self.models_dir = Path(self.test_dir) / "models"
        self.models_dir.mkdir()
        
        # Initialize DB
        database.init_db(self.db_path)
        
        # Create synthetic data (300 draws)
        self.create_synthetic_data()
        
        # Load dataframe
        with database.get_conn(self.db_path) as conn:
            # We need to manually load or use analyzer.load_dataframe logic
            # Simulating load_dataframe logic:
            data = conn.execute("SELECT * FROM draws ORDER BY draw_date").fetchall()
            cols = ["issue", "draw_date", "red1", "red2", "red3", "red4", "red5", "red6", "blue"]
            rows = []
            for d in data:
                rows.append([d["issue"], d["draw_date"], d["red1"], d["red2"], d["red3"], d["red4"], d["red5"], d["red6"], d["blue"]])
            self.df = pd.DataFrame(rows, columns=cols)
            
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_synthetic_data(self):
        draws = []
        start_date = pd.Timestamp("2024-01-01")
        for i in range(300):
            issue = f"2024{i:03d}"
            date = (start_date + pd.Timedelta(days=i)).date().isoformat()
            # Random valid lottery numbers
            reds = sorted(np.random.choice(range(1, 34), 6, replace=False))
            blue = np.random.randint(1, 17)
            draws.append(database.Draw(issue, date, list(reds), blue))
        
        with database.get_conn(self.db_path) as conn:
            database.upsert_draws(conn, draws)

    def test_nhits_training_and_predict(self):
        print("\n[Test] NHits Training")
        cfg = nhits_model.NHitsConfig(
            input_size=30,
            h=1,
            max_steps=10, # Very short training
            batch_size=16,
            learning_rate=1e-3
        )
        
        # Train
        model, cfg = nhits_model.train_nhits(self.df, cfg, save_dir=self.models_dir, fresh=True)
        self.assertIsNotNone(model)
        
        # Predict
        out = nhits_model.predict_nhits(model, cfg, self.df)
        self.assertIn("sum_pred", out)
        self.assertIn("blue_probs", out)
        
        # Check logic
        blue_probs = out["blue_probs"]
        self.assertEqual(len(blue_probs), 16)
        prob_sum = sum(p for n, p in blue_probs)
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_tft_training_and_predict(self):
        print("\n[Test] TFT Training")
        cfg = tft_model.TFTConfig(
            window=30,
            batch=16,
            epochs=2, # Very short
            d_model=16, # Small model
            nhead=2,
            layers=1,
            ff=32
        )
        
        model, cfg = tft_model.train_tft(self.df, cfg, save_dir=self.models_dir, fresh=True)
        self.assertIsNotNone(model)
        
        out = tft_model.predict_tft(model, cfg, self.df)
        self.assertIn("red", out)
        self.assertIn("blue", out)
        
        # Check logic
        red_preds = out["red"]
        self.assertEqual(len(red_preds), 6)
        
    def test_blender_logic(self):
        print("\n[Test] Blender Logic")
        # Create dummy predictions
        # Predictor 1
        def pred1(df):
            return {
                "blue": 5, 
                "blue_probs": [(5, 0.8), (1, 0.2)],
                "red": {1: [(1, 0.5)], 2: [(2, 0.5)]}
            }
        
        # Predictor 2
        def pred2(df):
             return {
                "blue": 10,
                "red_probs": [(1, 0.1) for _ in range(33)] # simplified
             }
             
        predictors = [pred1, pred2]
        
        # Check blend_blue_latest
        # Note: blend models usually need history features. 
        # _build_base_preds will run predictors on PAST data.
        # This might be slow if we run on self.df (300 rows).
        # We'll use a slice.
        
        small_df = self.df.iloc[-50:].reset_index(drop=True)
        
        # Run blender
        # Just ensure it doesn't crash and returns valid shape
        try:
            top_blue, prob = blender.blend_blue_latest(small_df, predictors)
            self.assertTrue(1 <= top_blue <= 16)
            self.assertTrue(0 <= prob <= 1.0)
        except Exception as e:
            self.fail(f"Blender crashed: {e}")

if __name__ == '__main__':
    unittest.main()
