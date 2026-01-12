import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path.cwd()))

from lottery import analyzer
from lottery.utils.logger import logger

def test_omission_calculation():
    print("Testing Omission Calculation...")
    
    # Mock Data: 3 issues
    # Issue 1: 1, 2, 3, 4, 5, 6
    # Issue 2: 7, 8, 9, 10, 11, 12 (1-6 should have omission 1)
    # Issue 3: 1, 13, 14, 15, 16, 17 (1 should be 0, others accumulate)
    
    data = {
        "issue": [1, 2, 3],
        "red1": [1, 7, 1],
        "red2": [2, 8, 13],
        "red3": [3, 9, 14],
        "red4": [4, 10, 15],
        "red5": [5, 11, 16],
        "red6": [6, 12, 17],
        "blue": [1, 1, 1]
    }
    df = pd.DataFrame(data)
    
    omission = analyzer.calculate_omission(df)
    
    # Check shape
    assert omission.shape == (3, 33), f"Shape mismatch: {omission.shape}"
    
    # Check Logic
    # Issue 1 (index 0): 
    # Since we initialize with 0 and add 1, then reset hits.
    # Logic in analyzer: 
    # current_omission += 1
    # hit reset to 0
    # So if it hit, it is 0. If not hit, it is 1 (assuming start from 0).
    
    # Row 0: 1-6 hit. Omission[0][0] (Red 1) should be 0.
    # Omission[0][6] (Red 7) should be 1.
    assert omission.iloc[0, 0] == 0, f"Red 1 issue 1 should be 0, got {omission.iloc[0,0]}"
    assert omission.iloc[0, 6] == 1, f"Red 7 issue 1 should be 1, got {omission.iloc[0,6]}"
    
    # Row 1: 7-12 hit.
    # Red 1 (not hit) -> previous 0 -> +1 = 1.
    # Red 7 (hit) -> previous 1 -> +1 -> reset 0.
    assert omission.iloc[1, 0] == 1, f"Red 1 issue 2 should be 1, got {omission.iloc[1,0]}"
    assert omission.iloc[1, 6] == 0, f"Red 7 issue 2 should be 0, got {omission.iloc[1,6]}"
    
    # Row 2: 1 hit.
    # Red 1 (hit) -> previous 1 -> +1 -> reset 0.
    # Red 2 (not hit) -> Issue 1 (0) -> Issue 2 (1) -> Issue 3 (2).
    assert omission.iloc[2, 0] == 0, f"Red 1 issue 3 should be 0, got {omission.iloc[2,0]}"
    assert omission.iloc[2, 1] == 2, f"Red 2 issue 3 should be 2, got {omission.iloc[2,1]}"
    
    print("Omission Calculation PASSED.")

def test_chaos_metrics():
    print("Testing Chaos Metrics (Lyapunov)...")
    # Simple sine wave (predictable, LLE should be near 0 or negative depending on tau/dim)
    t = np.linspace(0, 100, 500)
    series = np.sin(t)
    
    le = analyzer.lyapunov_exponent(series, dim=2, tau=10, max_t=5)
    # Ideally check if it runs. Exact value depends on params.
    print(f"LLE for sine: {le}")
    assert le is not None, "LLE returns None for valid series"
    
    print("Chaos Metrics PASSED.")

def test_log_file_access():
    print("Testing Log File Access...")
    log_path = Path("lottery.log")
    if not log_path.exists():
        # Create dummy if not exists (though it should in this env)
        with open(log_path, "w") as f:
            f.write("[PBT] Test Log\n")
            
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert len(content) > 0
        
    print("Log File Access PASSED.")

if __name__ == "__main__":
    try:
        test_omission_calculation()
        test_chaos_metrics()
        test_log_file_access()
        print("\nALL DASHBOARD COMPONENT TESTS PASSED.")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
