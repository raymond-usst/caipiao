import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import shutil
import logging

# Add project root to path
sys.path.append(str(Path.cwd()))

from lottery import features
from lottery.pipeline import train_model_task
from lottery.config import CatBoostConfig
from lottery.ml_model import CatBoostPredictor

# Setup basic logging to avoid clutter
logging.basicConfig(level=logging.INFO)

def test_polars_performance():
    print("--- Benchmark: Feature Engineering (Polars) ---")
    
    # generate 10k rows dummy data
    n = 10000
    print(f"Generating {n} rows of dummy data...")
    dates = pd.date_range("2000-01-01", periods=n)
    data = {
        "issue": range(1, n+1),
        "draw_date": dates,
        "red1": np.random.randint(1, 6, n),
        "red2": np.random.randint(7, 12, n),
        "red3": np.random.randint(13, 18, n),
        "red4": np.random.randint(19, 24, n),
        "red5": np.random.randint(25, 29, n),
        "red6": np.random.randint(30, 34, n),
        "blue": np.random.randint(1, 17, n)
    }
    df = pd.DataFrame(data)
    
    # Timed feature build
    start = time.time()
    res_df = features.build_features(df)
    end = time.time()
    
    elapsed = end - start
    print(f"Time taken: {elapsed:.4f} seconds ({len(df)} rows)")
    
    # Verification
    assert not res_df.empty
    expected_cols = ["ac", "span", "sum_tail", "odd_ratio", "omit_red_mean", "weekday_cos"]
    for c in expected_cols:
        assert c in res_df.columns, f"Missing column {c}"
        
    print(f"Output shape: {res_df.shape}")
    print("Polars Feature Engineering: PASSED")

def test_parallel_task_isolation():
    print("\n--- Test: Parallel Task Isolation ---")
    # Verify `train_model_task` works in isolation (it's the unit of work for parallel execution)
    
    n = 200
    dates = pd.date_range("2023-01-01", periods=n)
    df = pd.DataFrame({
        "issue": range(1, n+1),
        "draw_date": dates,
        "red1": np.random.randint(1, 6, n), # keep sorted roughly? no, just random
        "red2": np.random.randint(6, 11, n),
        "red3": np.random.randint(11, 16, n),
        "red4": np.random.randint(16, 21, n),
        "red5": np.random.randint(21, 26, n),
        "red6": np.random.randint(26, 34, n),
        "blue": np.random.randint(1, 17, n)
    })
    print(f"Mock Data Red1 Unique: {df['red1'].nunique()}")
    print(f"Mock Data Blue Unique: {df['blue'].nunique()}")
    
    # features
    df = features.build_features(df)
    
    # Config
    # We use a dummy model config. Assuming CatBoostConfig is compatible with TrainConfig structure or we use dict.
    # Actually `train_model_task` expects an object with enabled, resume, fresh attrs.
    
    class MockConfig:
        enabled = True
        resume = False
        fresh = True
        # Add minimal required for CatBoost (though it uses its own internal default mostly if not passed?)
        # CatBoostPredictor init takes config. 
        # Let's assume CatBoostPredictor handles minimal config or defaults.
        # We'll skip deep config details and assume defaults work if we pass enabled=True.
        # But CatBoostPredictor expects `config.epochs` etc? 
        # Let's inspect `ml_model.py` if needed.
        # CatBoostPredictor init: self.cfg = ...
        
    # We really need a valid config for CatBoostPredictor.
    # Config
    cfg = CatBoostConfig(enabled=True, fresh=True)
    
    # Output dir
    test_dir = Path("models_test_parallel")
    if test_dir.exists(): shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    try:
        success = train_model_task("CatBoost_Test", cfg, CatBoostPredictor, df, str(test_dir))
        assert success is True
        assert (test_dir / "catboost_red.cbm").exists()
        print("Parallel Task Execution: PASSED")
    finally:
        if test_dir.exists(): shutil.rmtree(test_dir)

if __name__ == "__main__":
    try:
        test_polars_performance()
        test_parallel_task_isolation()
        print("\nALL PERFORMANCE TESTS PASSED.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
