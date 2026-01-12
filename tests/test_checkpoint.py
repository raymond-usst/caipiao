import sys
from pathlib import Path
import pandas as pd
import torch
import shutil
import logging

# Add project root to path
sys.path.append(str(Path.cwd()))

from lottery.gnn_model import GNNPredictor
from lottery.config import GNNConfig
from lottery.utils.logger import logger

def test_gnn_checkpoint():
    print("Testing GNN Checkpoint...")
    
    # Setup
    save_dir = Path("models/GNN_Test")
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)
    
    # Mock Data
    dates = pd.date_range("2023-01-01", periods=50)
    df = pd.DataFrame({
        "issue": range(2023001, 2023051),
        "draw_date": dates,
        "red1": [1]*50, "red2": [2]*50, "red3": [3]*50, 
        "red4": [4]*50, "red5": [5]*50, "red6": [6]*50,
        "blue": [1]*50
    })
    
    cfg = GNNConfig(enabled=True, epochs=1, resume=True, fresh=False)
    
    # 1. First Run (Fresh because no checkpoint)
    print("--- Run 1 ---")
    predictor1 = GNNPredictor(cfg)
    # Patch save_dir inside the instance if possible, or we rely on the hardcoded "models/GNN" in gnn_model.py?
    # Ah, I hardcoded "models/GNN" in gnn_model.py. 
    # That makes testing hard without monkey-patching or modifying source.
    # I should have used a config variable or argument.
    
    # For this test, let's just run it. It will write to "models/GNN".
    # We will back up "models/GNN" if it exists.
    real_dir = Path("models/GNN")
    backup_dir = Path("models/GNN_Backup")
    if real_dir.exists():
        real_dir.rename(backup_dir)
    
    try:
        predictor1.train(df) # Should Init Fresh
        w1 = predictor1.model.out.weight.clone()
        predictor1.save(str(real_dir)) # Save manually to ensure structure
        
        # 2. Second Run (Resume)
        print("--- Run 2 ---")
        predictor2 = GNNPredictor(cfg)
        predictor2.train(df) # Should Resume
        w2 = predictor2.model.out.weight
        
        # Check if w2 is close to w1 (it trained more, so it changed, but initialization should allow continuity)
        # Actually, if we resume, we load w1. Then we train for 1 epoch.
        # If we didn't resume (fresh init), w2 would be random and very different.
        # But we can verify explicitly if the "Resumed" log message appears or check internal flags.
        # Or better: Check if `predictor2.load` was successful.
        
        # Simpler check: modifying weights manually and seeing if they persist.
        predictor1.model.out.weight.data.fill_(1.0)
        predictor1.save(str(real_dir))
        
        predictor3 = GNNPredictor(cfg)
        predictor3.train(df)
        w3 = predictor3.model.out.weight
        
        if torch.allclose(w3, torch.tensor(1.0).to(w3.device)):
            print("SUCCESS: GNN loads checkpoint correctly.")
        else:
            print("FAILURE: GNN did not load checkpoint (weights not 1.0).")
            print(w3[0,:5])
            
    finally:
        # Cleanup
        if real_dir.exists():
            shutil.rmtree(real_dir)
        if backup_dir.exists():
            backup_dir.rename(real_dir)

if __name__ == "__main__":
    test_gnn_checkpoint()
