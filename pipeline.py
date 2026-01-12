from pathlib import Path
from typing import Dict, List, Any, Type
import pandas as pd
import concurrent.futures
from lottery.utils.logger import logger
from lottery.config import TrainAllConfig
from lottery.engine.predictor import BasePredictor
from lottery import database, analyzer, blender

# Import Predictors
from lottery.ml_model import CatBoostPredictor
from lottery.seq_model import SeqPredictor
from lottery.tft_model import TFTPredictor
from lottery.nhits_model import NHitsPredictor
from lottery.timesnet_model import TimesNetPredictor
from lottery.prophet_model import ProphetPredictor
from lottery.gnn_model import GNNPredictor
from lottery.rl_model import RLPredictor
from lottery.esn_model import ESNPredictor
from lottery.bnn_model import BNNPredictor
from lottery.meta_model import MetaLearningPredictor

def train_model_task(name: str, config: Any, PredictorClass: Type[BasePredictor], df: pd.DataFrame, models_dir: str) -> bool:
    """
    Task to be executed in a separate process.
    Returns True if model is ready (trained or loaded), False if failed.
    """
    if not config.enabled:
        return False
        
    try:
        # Re-init logger in subprocess might be needed if not automatically handled, 
        # but standard logging checks locks. Simple print might be safer or just rely on main process logging tasks completion.
        # But we want to see progress.
        
        predictor = PredictorClass(config)
        loaded = False
        if config.resume and not config.fresh:
             loaded = predictor.load(models_dir)
        
        if not loaded or config.fresh:
             logger.info(f"[Parallel] Training {name}...")
             predictor.train(df)
             predictor.save(models_dir)
             logger.info(f"[Parallel] {name} trained and saved.")
        else:
             logger.info(f"[Parallel] {name} loaded from checkpoint.")
             
        return True
    except Exception as e:
        logger.error(f"[Parallel] Failed {name}: {e}")
        return False

def run_pipeline(cfg: TrainAllConfig) -> None:
    """
    Execute the training and blending pipeline based on the configuration.
    """
    logger.info("Starting training pipeline (Parallel)...")
    
    # 1. Database & Data Loading
    db_path = Path(cfg.db.path)
    if cfg.sync:
        logger.info("Syncing database...")
        from lottery import scraper
        database.init_db(db_path)
        all_draws = scraper.fetch_all_draws()
        with database.get_conn(db_path) as conn:
            existing = {row["issue"] for row in conn.execute("SELECT issue FROM draws")}
            to_insert = scraper.filter_new_draws(all_draws, existing)
            if not existing: to_insert = all_draws
            if to_insert:
                affected = database.upsert_draws(conn, to_insert)
                logger.info(f"Synced {affected} new draws.")
            else:
                logger.info("Database is up to date.")

    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=cfg.recent)
    
    if df.empty:
        logger.error("Dataset is empty. Exiting.")
        return

    logger.info(f"Loaded {len(df)} samples.")
    
    # 2. Parallel Model Training
    models_dir = "models"
    Path(models_dir).mkdir(exist_ok=True)
    
    # Define tasks
    # (name, config, Class)
    tasks = [
        ("CatBoost", cfg.cat, CatBoostPredictor),
        ("SeqModel", cfg.seq, SeqPredictor),
        ("TFT", cfg.tft, TFTPredictor),
        ("NHits", cfg.nhits, NHitsPredictor),
        ("TimesNet", cfg.timesnet, TimesNetPredictor),
        ("Prophet", cfg.prophet, ProphetPredictor),
        ("GNN", cfg.gnn, GNNPredictor),
        ("RL", cfg.rl, RLPredictor),
        ("ESN", cfg.esn, ESNPredictor),
        ("BNN", cfg.bnn, BNNPredictor),
        ("Meta", cfg.meta, MetaLearningPredictor),
    ]
    
    # Filter enabled
    tasks = [t for t in tasks if t[1].enabled]
    
    failed_models = set()
    
    # Use ProcessPoolExecutor
    # Note: df is passed to all. 
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(tasks), 4)) as executor:
        future_to_name = {
            executor.submit(train_model_task, name, conf, cls, df, models_dir): name
            for name, conf, cls in tasks
        }
        
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                success = future.result()
                if not success:
                    failed_models.add(name)
            except Exception as e:
                logger.error(f"Task {name} generated an exception: {e}")
                failed_models.add(name)
                
    # 3. Reload Models for Blending
    # We need to instantiate them in the main process to use them for prediction blending
    predictors: List[Any] = []
    
    for name, conf, cls in tasks:
        if name in failed_models:
            continue
            
        try:
            p = cls(conf)
            # Force load (since we expect it to be saved by subprocess)
            if p.load(models_dir):
                # Wrapper
                def pred_wrapper(hist_df, _p=p): 
                    return _p.predict(hist_df)
                predictors.append(pred_wrapper)
            else:
                logger.warning(f"Could not load {name} for blending even though training reported success.")
        except Exception as e:
            logger.error(f"Error loading {name} for blending: {e}")

    # 4. Blending
    if cfg.blend.enabled:
        if len(predictors) < 2:
            logger.warning("Not enough models for blending (need at least 2). Skipping blend.")
            if predictors:
                result_summary(predictors, df)
            return
            
        logger.info(f"Blending {len(predictors)} models...")
        
        # Blue Ball
        avg_acc_b, folds_b = blender.rolling_blend_blue(df, predictors, cfg=cfg.blend)
        fused_num_b, fused_prob_b = blender.blend_blue_latest(df, predictors)
        logger.info(f"[Blend] Blue Top1 Acc: {avg_acc_b:.3f}, Next Prediction: {fused_num_b} (prob={fused_prob_b:.3f})")
        
        # Sum
        avg_mae_s, folds_s = blender.rolling_blend_sum(df, predictors, cfg=cfg.blend)
        fused_sum = blender.blend_sum_latest(df, predictors)
        logger.info(f"[Blend] Sum MAE: {avg_mae_s:.3f}, Next Sum: {fused_sum:.2f}")
        
        # Red Positions
        avg_acc_r, folds_r = blender.rolling_blend_red(df, predictors, cfg=cfg.blend)
        fused_red = blender.blend_red_latest(df, predictors)
        logger.info(f"[Blend] Red Pos Acc: {avg_acc_r:.3f}, Next Red: {fused_red}")
        
        print("\n" + "="*30)
        print(f" FINAL PREDICTION ")
        print(f" RED : {fused_red}")
        print(f" BLUE: {fused_num_b}")
        print("="*30 + "\n")

def result_summary(predictors, df):
    # Just print latest prediction from first model if blend skipped
    if not predictors: return
    p = predictors[0]
    res = p(df)
    logger.info(f"Single model prediction: {res}")
