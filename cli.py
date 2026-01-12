import argparse
import sys
from pathlib import Path
from lottery.utils.logger import logger
from lottery.config import load_config
from lottery import database, analyzer, scraper

# Command Handlers
def cmd_sync(args, cfg):
    logger.info(f"Syncing data... (recent={args.recent})")
    db_path = Path(cfg.db.path)
    database.init_db(db_path)
    
    all_draws = scraper.fetch_all_draws()
    with database.get_conn(db_path) as conn:
        existing = {row["issue"] for row in conn.execute("SELECT issue FROM draws")}
        to_insert = scraper.filter_new_draws(all_draws, existing)
        if not existing: to_insert = all_draws
    
    if to_insert:
        with database.get_conn(db_path) as conn:
            affected = database.upsert_draws(conn, to_insert)
            logger.info(f"Synced {affected} new draws.")
    else:
        logger.info("Database is up to date.")

def cmd_analyze(args, cfg):
    db_path = Path(cfg.db.path)
    logger.info(f"Analyzing data from {db_path}...")
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if df.empty:
        logger.warning("No data found.")
        return
    analyzer.show_basic_stats(df)
    analyzer.analyze_correlations(df)

def cmd_train_all(args, cfg):
    from lottery.pipeline import run_pipeline
    # Override config with CLI args if necessary (optional, e.g. force fresh)
    # For now, we rely on the config file, but we can set overrides here.
    if args.fresh:
        cfg.fresh = True # Assuming global fresh override logic exists or we patch configs
        # Patching all model configs to fresh=True is tedious, better handled by config loader 
        # or by passing overrides to run_pipeline.
        # For simplicity, let's update the unified config object:
        cfg.cat.fresh = True
        cfg.seq.fresh = True
        cfg.tft.fresh = True
        cfg.nhits.fresh = True
        cfg.timesnet.fresh = True
        cfg.prophet.fresh = True
        cfg.sync = args.sync # CLI override
    
    run_pipeline(cfg)

def cmd_backtest(args, cfg):
    from lottery.engine.backtest import RollingBacktester
    from lottery.ml_model import CatBoostPredictor
    from lottery.seq_model import SeqPredictor
    from lottery.tft_model import TFTPredictor
    from lottery.nhits_model import NHitsPredictor
    from lottery.timesnet_model import TimesNetPredictor
    from lottery.prophet_model import ProphetPredictor
    from lottery.gnn_model import GNNPredictor
    from lottery.rl_model import RLPredictor

    # Select model class based on args
    name_map = {
        "cat": (CatBoostPredictor, cfg.cat),
        "seq": (SeqPredictor, cfg.seq),
        "tft": (TFTPredictor, cfg.tft),
        "nhits": (NHitsPredictor, cfg.nhits),
        "timesnet": (TimesNetPredictor, cfg.timesnet),
        "prophet": (ProphetPredictor, cfg.prophet),
        "gnn": (GNNPredictor, cfg.gnn),
        "rl": (RLPredictor, cfg.rl),
    }
    
    if args.model not in name_map:
        logger.error(f"Unknown model: {args.model}")
        return

    cls, model_cfg = name_map[args.model]
    
    # Load data
    db_path = Path(cfg.db.path)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    
    if df.empty:
        logger.error("No data for backtest.")
        return
        
    tester = RollingBacktester(cls, model_cfg, retrain_interval=args.interval)
    tester.run(df, start_ratio=0.8)


def main():
    parser = argparse.ArgumentParser(description="Lottery Prediction CLI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Sync
    p_sync = subparsers.add_parser("sync", help="Sync data from web")
    p_sync.add_argument("--recent", type=int, default=100, help="Number of recent draws to check/fetch")

    # Analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze data stats")
    p_analyze.add_argument("--recent", type=int, default=300, help="Number of recent draws to analyze")

    # Train All
    p_train = subparsers.add_parser("train-all", help="Train all models and blend")
    p_train.add_argument("--fresh", action="store_true", help="Force retrain all models")
    p_train.add_argument("--sync", action="store_true", help="Sync data before training")

    # Backtest
    p_back = subparsers.add_parser("backtest", help="Run rolling backtest")
    p_back.add_argument("--model", type=str, required=True, choices=["cat", "seq", "tft", "nhits", "timesnet", "prophet", "gnn", "rl"], help="Model to backtest")
    p_back.add_argument("--interval", type=int, default=1, help="Retraining interval (samples)")
    p_back.add_argument("--recent", type=int, default=100, help="Data size to use")
    
    # Dashboard
    p_dash = subparsers.add_parser("dashboard", help="Launch Streamlit Dashboard")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load Config
    try:
        cfg = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Dispatch
    if args.command == "sync":
        cmd_sync(args, cfg)
    elif args.command == "analyze":
        cmd_analyze(args, cfg)
    elif args.command == "train-all":
        cmd_train_all(args, cfg)
    elif args.command == "predict":
        cmd_predict(args, cfg)
    elif args.command == "backtest":
        cmd_backtest(args, cfg)
    elif args.command == "dashboard":
        import sys
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", "dashboard.py"]
        sys.exit(stcli.main())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
