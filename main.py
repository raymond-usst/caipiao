from __future__ import annotations

import argparse
import sys
import os
import random
from pathlib import Path
import pandas as pd
import numpy as np

# 统一控制台编码为 UTF-8，避免中文输出乱码（Windows PowerShell 常见）
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Patch for tensorboard/numpy compatibility
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

import logging
# Suppress all Lightning logs including "Seed set to 42"
class SeedFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Seed set to" in msg:
            return False
        return True

# Initialize logging to ensure handlers exist, then attach filter to ALL handlers
logging.basicConfig(level=logging.ERROR)
seed_filter = SeedFilter()
logging.getLogger().addFilter(seed_filter)
for handler in logging.getLogger().handlers:
    handler.addFilter(seed_filter)

# Also explicitly mute known noisy loggers
for name in ["lightning", "pytorch_lightning", "lightning.fabric", "lightning.pytorch", "neuralforecast", "cmdstanpy"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.addFilter(seed_filter)

from lottery import analyzer, database, scraper, blender
from lottery import ml_model, seq_model, tft_model, nhits_model, timesnet_model, prophet_model, validation, odd_model, sum_model, pbt


def cmd_sync(args) -> None:
    db_path = Path(args.db)
    print(f"[sync] 使用数据库: {db_path}")
    database.init_db(db_path)

    all_draws = scraper.fetch_all_draws()
    print(f"[sync] 抓取到 {len(all_draws)} 期历史数据")

    with database.get_conn(db_path) as conn:
        existing = {row["issue"] for row in conn.execute("SELECT issue FROM draws")}
        to_insert = scraper.filter_new_draws(all_draws, existing)
        # 若已有数据为空，则全量写入
        if not existing:
            to_insert = all_draws

        if not to_insert:
            print("[sync] 数据库已是最新，无需更新")
            return

        affected = database.upsert_draws(conn, to_insert)
        latest_issue = max(d.issue for d in to_insert)
        print(f"[sync] 新增/更新 {affected} 期，最新期号 {latest_issue}")


def cmd_analyze(args) -> None:
    db_path = Path(args.db)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)

    if df.empty:
        print("数据库为空，请先运行 sync 同步数据。")
        sys.exit(1)

    report = analyzer.analyze(df, entropy_window=args.entropy_window)

    print(f"[analyze] 样本期数: {report['count']}, 最新期号: {report['latest_issue']}")
    print(f"[analyze] 高频红球: {report['hot_red']}  冷门红球: {report['cold_red']}")
    print(f"[analyze] 高频蓝球: {report['hot_blue']}  冷门蓝球: {report['cold_blue']}")
    if report["entropy_recent"] is not None:
        print(f"[analyze] 近期（窗口 {args.entropy_window}）熵: {report['entropy_recent']:.4f}")
    if report["lyapunov"] is not None:
        print(f"[analyze] 近似最大李雅普诺夫指数: {report['lyapunov']:.4f} (越大越具混沌)")
    stats = report.get("basic_stats", {})
    if stats:
        print(
            f"[analyze] 基础统计: 红均值 {stats['red_mean']:.2f}±{stats['red_std']:.2f}, "
            f"蓝均值 {stats['blue_mean']:.2f}±{stats['blue_std']:.2f}, "
            f"和值均值 {stats['sum_mean']:.2f}±{stats['sum_std']:.2f}"
        )
    chi = report.get("chi_square", {})
    if chi:
        print(f"[analyze] 卡方检验(红): stat={chi['red']['stat']:.2f}, dof={chi['red']['dof']} (均匀性偏离越大stat越大)")
        print(f"[analyze] 卡方检验(蓝): stat={chi['blue']['stat']:.2f}, dof={chi['blue']['dof']}")
    runs = report.get("runs_test_sum")
    if runs:
        print(f"[analyze] 游程检验(和值序列): runs={runs['runs']}, z={runs['z']:.3f}, p≈{runs['p']:.3f} (p小于0.05则拒绝独立同分布假设)")
    corr_dim = report.get("corr_dimension")
    if corr_dim is not None:
        print(f"[analyze] 相关维(简化GP): {corr_dim:.3f}")
    fnn = report.get("fnn", {})
    if fnn:
        msg = ", ".join(f"m={m}: {v:.2%}" for m, v in sorted(fnn.items()))
        print(f"[analyze] 虚假最近邻比例: {msg} (越低越能说明嵌入维度充分)")
    sig = report.get("significant", {})
    if sig:
        red_hot = sig.get("red", {}).get("hot", [])
        red_cold = sig.get("red", {}).get("cold", [])
        blue_hot = sig.get("blue", {}).get("hot", [])
        blue_cold = sig.get("blue", {}).get("cold", [])
        if red_hot or red_cold:
            print(f"[analyze] 显著热红号(z>=2): {red_hot}  显著冷红号(z<=-2): {red_cold}")
        if blue_hot or blue_cold:
            print(f"[analyze] 显著热蓝号(z>=2): {blue_hot}  显著冷蓝号(z<=-2): {blue_cold}")
    ac = report.get("autocorr", {})
    if ac:
        show_lags = [lag for lag in sorted(ac.keys()) if lag <= 5]
        vals = ", ".join(f"lag{l}={ac[l]:.3f}" for l in show_lags)
        print(f"[analyze] 和值序列自相关: {vals}")
    omission = report.get("omission", {})
    if omission:
        red_om = omission.get("red", {})
        blue_om = omission.get("blue", {})
        red_sorted = sorted(red_om.items(), key=lambda kv: kv[1]["current"], reverse=True)[:3]
        blue_sorted = sorted(blue_om.items(), key=lambda kv: kv[1]["current"], reverse=True)[:3]
        red_msg = ", ".join(f"{n}(当前{v['current']}, 历史Max{v['max']})" for n, v in red_sorted)
        blue_msg = ", ".join(f"{n}(当前{v['current']}, 历史Max{v['max']})" for n, v in blue_sorted)
        print(f"[analyze] 红球遗漏Top3: {red_msg}")
        print(f"[analyze] 蓝球遗漏Top3: {blue_msg}")
    periodic = report.get("periodic", {})
    if periodic:
        red_all = periodic.get("red", {}).get("all", [])
        blue_all = periodic.get("blue", {}).get("all", [])
        red_p = periodic.get("red", {}).get("periodic", [])[:3]
        blue_p = periodic.get("blue", {}).get("periodic", [])[:3]
        if red_p:
            msg = ", ".join(f"{x['num']}(均间隔{round(x['mean_gap'],1)}±{round(x['std_gap'],1)}, CV={x['cv']:.2f})" for x in red_p)
            print(f"[analyze] 可疑准时出现的红号: {msg}")
        if blue_p:
            msg = ", ".join(f"{x['num']}(均间隔{round(x['mean_gap'],1)}±{round(x['std_gap'],1)}, CV={x['cv']:.2f})" for x in blue_p)
            print(f"[analyze] 可疑准时出现的蓝号: {msg}")
        if red_all:
            top = red_all[:5]
            msg = "; ".join(f"{x['num']}:均{round(x['mean_gap'],1)}±{round(x['std_gap'],1)},CV={x['cv']:.2f},min={x['min_gap']},max={x['max_gap']}" for x in top)
            print(f"[analyze] 红球间隔统计CV最小Top5: {msg}")
        if blue_all:
            top = blue_all[:5]
            msg = "; ".join(f"{x['num']}:均{round(x['mean_gap'],1)}±{round(x['std_gap'],1)},CV={x['cv']:.2f},min={x['min_gap']},max={x['max_gap']}" for x in top)
            print(f"[analyze] 蓝球间隔统计CV最小Top5: {msg}")
    chaos = report.get("chaos", {})
    if chaos:
        if chaos.get("corr_dim"):
            cd = chaos["corr_dim"]
            msg = "; ".join(f"m{m}:{cd[m]:.2f}" for m in sorted(cd.keys()))
            print(f"[analyze] 相关维数估计: {msg}")
        if chaos.get("fnn"):
            fnn = chaos["fnn"]
            msg = "; ".join(f"m{m}->{m+1}:{fnn[m]:.1f}%" for m in sorted(fnn.keys()))
            print(f"[analyze] 假最近邻比例: {msg}")
        if chaos.get("recur"):
            r = chaos["recur"]
            print(f"[analyze] 复现率RR={r.get('rr', float('nan')):.4f}, DET={r.get('det', float('nan')):.4f} (eps={r.get('eps', float('nan')):.4f})")
    rules = report.get("rules", [])
    if rules:
        top_rules = rules[:5]
        formatted = "; ".join(
            f"{r['antecedent']} -> {r['consequent']} (sup={r['support']:.3f}, conf={r['confidence']:.2f}, lift={r['lift']:.2f})"
            for r in top_rules
        )
        print(f"[analyze] Apriori 关联规则Top5: {formatted}")
    else:
        print("[analyze] Apriori 关联规则: 暂无满足阈值的规则 (默认 sup>=0.01, conf>=0.2)")
    suggestion = report["suggestion"]
    print(f"[analyze] 基于频率与混沌检验的推荐组合: 红 {suggestion['reds']} + 蓝 {suggestion['blue']}")


def cmd_predict(args) -> None:
    db_path = Path(args.db)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if df.empty or len(df) < args.window + 1:
        print("样本不足，无法训练预测模型。")
        sys.exit(1)
    print(f"[predict][blend] 使用最近 {len(df)} 期数据，窗口 {args.window}，CatBoost 迭代 {args.iter}")

    base_predictors = []

    # CatBoost 基础模型（可选贝叶斯调参）
    cat_iter = args.iter
    cat_depth = args.depth
    cat_lr = args.lr
    if getattr(args, "bayes_cat", False):
        try:
            best_est, best_params, best_score = ml_model.bayes_optimize_catboost(
                df, window=args.window, n_iter=args.bayes_cat_calls, cv_splits=args.bayes_cat_cv, save_dir="models"
            )
            cat_depth = int(best_params.get("depth", cat_depth))
            cat_lr = float(best_params.get("learning_rate", cat_lr))
            cat_iter = int(best_params.get("iterations", cat_iter))
            print(f"[bayes-cat] 最优参数: depth={cat_depth}, lr={cat_lr:.4f}, iter={cat_iter}, score={best_score:.4f}")
        except Exception as e:
            print(f"[bayes-cat][warn] 调参失败，回退手动参数: {e}")
    resume = not args.cat_no_resume
    fresh = args.cat_fresh
    cat_models = ml_model.train_models(
        df,
        window=args.window,
        iterations=cat_iter,
        depth=cat_depth,
        learning_rate=cat_lr,
        save_dir="models",
        resume=resume,
        fresh=fresh,
    )
    cat_preds = ml_model.predict_next(cat_models, df, top_k=args.topk)
    for pos, items in cat_preds["red"].items():
        print(f"[predict][catboost] 位置{pos} Top{args.topk}: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
    print(f"[predict][catboost] 蓝球 Top{args.topk}: " + ", ".join([f"{n}({p:.3f})" for n, p in cat_preds['blue']]))

    def pred_cat(hist_df):
        return ml_model.predict_next(cat_models, hist_df, top_k=1)

    base_predictors.append(pred_cat)

    # 频率基线（不训练）
    def pred_freq(hist_df):
        cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
        data = hist_df[cols].to_numpy(dtype=int)
        reds = data[:, :6].ravel()
        blues = data[:, 6]
        red_freq = np.bincount(reds, minlength=34)
        blue_freq = np.bincount(blues, minlength=17)
        red_top = np.argsort(red_freq[1:])[::-1][:6] + 1
        blue_top = np.argsort(blue_freq[1:])[::-1][:1] + 1
        red_preds = {i + 1: [(int(n), 1.0 / len(red_top))] for i, n in enumerate(red_top[:6])}
        blue_preds = [(int(blue_top[0]), 1.0)]
        return {"red": red_preds, "blue": blue_preds, "sum_pred": float(data[-1, :].sum())}

    base_predictors.append(pred_freq)

    # 随机基线（不训练）：均匀采样 6 个红球 + 1 个蓝球，便于评估模型相对优势
    def pred_random(hist_df):
        rng = np.random.default_rng(len(hist_df))  # 使同样窗口长度可复现
        red_nums = rng.choice(np.arange(1, 34), size=6, replace=False)
        blue_num = int(rng.integers(1, 17))
        red_preds = {i + 1: [(int(n), 1.0 / 6)] for i, n in enumerate(red_nums)}
        blue_preds = [(blue_num, 1.0)]
        # 使用最近一期和值作为简单占位
        last_sum = float(hist_df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].iloc[-1].sum())
        return {"red": red_preds, "blue": blue_preds, "sum_pred": last_sum}

    base_predictors.append(pred_random)

    # Transformer
    try:
        # 默认超参
        seq_window = args.window
        seq_bs = 64
        seq_epochs = 10
        seq_lr = 5e-4
        seq_dm = 96
        seq_nh = 4
        seq_layers = 3
        seq_ff = 192
        seq_dr = 0.1
        # 可选贝叶斯调参
        if getattr(args, "bayes_seq", False):
            try:
                best_params, best_loss = seq_model.bayes_optimize_seq(
                    df, recent=max(200, args.recent or 0), n_iter=args.bayes_seq_calls
                )
                seq_window = int(best_params["window"])
                seq_bs = int(best_params["batch_size"])
                seq_lr = float(best_params["lr"])
                seq_dm = int(best_params["d_model"])
                seq_nh = int(best_params["nhead"])
                seq_layers = int(best_params["num_layers"])
                seq_ff = int(best_params["ff"])
                seq_dr = float(best_params["dropout"])
                print(f"[bayes-seq] 最优参数: {best_params}, loss={best_loss:.4f}")
            except Exception as e:
                print(f"[bayes-seq][warn] 调参失败，回退默认: {e}")
        seq_cfg = seq_model.TrainConfig(
            window=seq_window,
            batch_size=seq_bs,
            epochs=seq_epochs,
            lr=seq_lr,
            d_model=seq_dm,
            nhead=seq_nh,
            num_layers=seq_layers,
            ff=seq_ff,
            dropout=seq_dr,
            topk=1,
        )
        seq_resume = not getattr(args, "seq_no_resume", False)
        seq_fresh = getattr(args, "seq_fresh", False)
        seq_m, seq_cfg = seq_model.train_seq_model(
            df.tail(max(800, seq_window + 5)),
            seq_cfg,
            save_dir="models",
            resume=seq_resume,
            fresh=seq_fresh,
        )

        def pred_seq(hist_df):
            return seq_model.predict_seq(seq_m, seq_cfg, hist_df)

        base_predictors.append(pred_seq)
        print("[predict][seq] added to blend")

        if getattr(args, "seq_backtest", False):
            try:
                bt = seq_model.backtest_seq_model(seq_m, seq_cfg, df, batch_size=args.seq_backtest_batch)
                print(f"[predict][seq][backtest] red_top1={bt['red_top1']:.3f}, blue_top1={bt['blue_top1']:.3f}, samples={bt['samples']}")
            except Exception as e:
                print(f"[predict][seq][backtest][warn] {e}")
    except Exception as e:
        print(f"[predict][seq][warn] 跳过: {e}")

    # TFT
    try:
        tft_window = args.window
        tft_bs = 64
        tft_epochs = 10
        tft_lr = 5e-4
        tft_dm = 128
        tft_nh = 4
        tft_layers = 3
        tft_ff = 256
        tft_dr = 0.1
        tft_fw = 50
        tft_ew = 50
        best_params = None
        best_loss = 1e9
        if getattr(args, "bayes_tft", False):
            try:
                bp, bl = tft_model.bayes_optimize_tft(
                    df, recent=max(args.bayes_tft_recent, args.recent or 0), n_iter=args.bayes_tft_calls
                )
                best_params, best_loss = bp, bl
                print(f"[bayes-tft] 最优参数: {bp}, loss={bl:.4f}")
            except Exception as e:
                print(f"[bayes-tft][warn] 调参失败，回退默认: {e}")
        if getattr(args, "tft_rand", False):
            try:
                rp, rl = tft_model.random_optimize_tft(
                    df, recent=max(args.bayes_tft_recent, args.recent or 0), samples=args.tft_rand_samples
                )
                print(f"[rand-tft] 最优参数: {rp}, loss={rl:.4f}")
                if rl < best_loss:
                    best_params, best_loss = rp, rl
            except Exception as e:
                print(f"[rand-tft][warn] 调参失败，忽略: {e}")
        if best_params:
            tft_window = int(best_params["window"])
            tft_bs = int(best_params["batch"])
            tft_lr = float(best_params["lr"])
            tft_dm = int(best_params["d_model"])
            tft_nh = int(best_params["nhead"])
            tft_layers = int(best_params["layers"])
            tft_ff = int(best_params["ff"])
            tft_dr = float(best_params["dropout"])
        tft_cfg = tft_model.TFTConfig(
            window=tft_window,
            batch=tft_bs,
            epochs=tft_epochs,
            lr=tft_lr,
            d_model=tft_dm,
            nhead=tft_nh,
            layers=tft_layers,
            ff=tft_ff,
            dropout=tft_dr,
            topk=1,
            freq_window=tft_fw,
            entropy_window=tft_ew,
        )
        tft_resume = not getattr(args, "tft_no_resume", False)
        tft_fresh = getattr(args, "tft_fresh", False)
        tft_m, tft_cfg = tft_model.train_tft(
            df.tail(max(800, tft_window + 5)),
            tft_cfg,
            save_dir="models",
            resume=tft_resume,
            fresh=tft_fresh,
        )

        def pred_tft(hist_df):
            return tft_model.predict_tft(tft_m, tft_cfg, hist_df)

        base_predictors.append(pred_tft)
        print("[predict][tft] added to blend")

        if getattr(args, "tft_backtest", False):
            try:
                bt = tft_model.backtest_tft_model(tft_m, tft_cfg, df, batch_size=args.tft_backtest_batch)
                print(f"[predict][tft][backtest] red_top1={bt['red_top1']:.3f}, blue_top1={bt['blue_top1']:.3f}, samples={bt['samples']}")
            except Exception as e:
                print(f"[predict][tft][backtest][warn] {e}")
    except Exception as e:
        print(f"[predict][tft][warn] 跳过: {e}")

    # N-HiTS（和值/蓝球）
    try:
        nh_input = min(90, len(df) // 2)
        nh_lr = 5e-4
        nh_ms = 80
        nh_bs = 32
        if getattr(args, "bayes_nhits", False):
            try:
                best_params, best_loss = nhits_model.bayes_optimize_nhits(
                    df, recent=args.bayes_nhits_recent, n_iter=args.bayes_nhits_calls
                )
                nh_input = int(best_params["input_size"])
                nh_lr = float(best_params["learning_rate"])
                nh_ms = int(best_params["max_steps"])
                nh_bs = int(best_params["batch_size"])
                print(f"[bayes-nhits] 最优参数: {best_params}, loss={best_loss:.4f}")
            except Exception as e:
                print(f"[bayes-nhits][warn] 调参失败，回退默认: {e}")
        nh_cfg = nhits_model.NHitsConfig(
            input_size=nh_input,
            h=1,
            learning_rate=nh_lr,
            max_steps=nh_ms,
            batch_size=nh_bs,
            valid_size=0.1,
            early_stop_patience_steps=5,
        )
        nh_resume = not getattr(args, "nhits_no_resume", False)
        nh_fresh = getattr(args, "nhits_fresh", False)
        nh_m, nh_cfg = nhits_model.train_nhits(df, nh_cfg, save_dir="models", resume=nh_resume, fresh=nh_fresh)

        def pred_nh(hist_df):
            out = nhits_model.predict_nhits(nh_m, nh_cfg, hist_df)
            # 将全局红球概率复制到各位置，蓝球直接用 blue_probs
            if out.get("red_probs"):
                red_probs = out["red_probs"]
                out["red"] = {pos: red_probs for pos in range(1, 7)}
            if out.get("blue_probs"):
                out["blue"] = out["blue_probs"]
            return out

        base_predictors.append(pred_nh)
        if getattr(args, "nhits_backtest", False):
            try:
                bt = nhits_model.backtest_nhits_model(
                    nh_m,
                    nh_cfg,
                    df,
                    max_samples=getattr(args, "nhits_backtest_samples", None),
                )
                print(
                    f"[predict][nhits][backtest] mae_sum={bt['mae_sum']:.3f}, mae_blue={bt['mae_blue']:.3f}, blue_top1={bt['blue_top1']:.3f}, samples={bt['samples']}"
                )
            except Exception as e:
                print(f"[predict][nhits][backtest][warn] {e}")
        print("[predict][nhits] added to blend")
    except Exception as e:
        print(f"[predict][nhits][warn] 跳过: {e}")

    # TimesNet（和值/蓝球）
    try:
        tn_input = min(150, len(df) // 2)
        tn_lr = 5e-4
        tn_ms = 100
        tn_bs = 32
        tn_hidden = 64
        tn_topk = 5
        tn_dr = 0.1
        best_params = None
        best_loss = 1e9
        if getattr(args, "bayes_timesnet", False):
            try:
                bp, bl = timesnet_model.bayes_optimize_timesnet(
                    df, recent=args.bayes_timesnet_recent, n_iter=args.bayes_timesnet_calls
                )
                best_params, best_loss = bp, bl
                print(f"[bayes-timesnet] 最优参数: {bp}, loss={bl:.4f}")
            except Exception as e:
                print(f"[bayes-timesnet][warn] 调参失败，回退默认: {e}")
        if getattr(args, "timesnet_rand", False):
            try:
                rp, rl = timesnet_model.random_optimize_timesnet(
                    df, recent=args.bayes_timesnet_recent, samples=args.timesnet_rand_samples
                )
                print(f"[rand-timesnet] 最优参数: {rp}, loss={rl:.4f}")
                if rl < best_loss:
                    best_params, best_loss = rp, rl
            except Exception as e:
                print(f"[rand-timesnet][warn] 调参失败，忽略: {e}")
        if best_params:
            tn_input = int(best_params["input_size"])
            tn_hidden = int(best_params["hidden_size"])
            tn_topk = int(best_params["top_k"])
            tn_dr = float(best_params["dropout"])
            tn_lr = float(best_params["learning_rate"])
            tn_ms = int(best_params["max_steps"])
            tn_bs = int(best_params["batch_size"])
        tn_cfg = timesnet_model.TimesNetConfig(
            input_size=tn_input,
            h=1,
            hidden_size=tn_hidden,
            top_k=tn_topk,
            dropout=tn_dr,
            learning_rate=tn_lr,
            max_steps=tn_ms,
            batch_size=tn_bs,
            valid_size=0.1,
        )
        tn_resume = not getattr(args, "timesnet_no_resume", False)
        tn_fresh = getattr(args, "timesnet_fresh", False)
        tn_m, tn_cfg = timesnet_model.train_timesnet(df, tn_cfg, save_dir="models", resume=tn_resume, fresh=tn_fresh)

        def pred_tn(hist_df):
            out = timesnet_model.predict_timesnet(tn_m, hist_df)
            if out.get("red_probs"):
                red_probs = out["red_probs"]
                out["red"] = {pos: red_probs for pos in range(1, 7)}
            if out.get("blue_probs"):
                out["blue"] = out["blue_probs"]
            return out

        base_predictors.append(pred_tn)
        print("[predict][timesnet] added to blend")
    except Exception as e:
        print(f"[predict][timesnet][warn] 跳过: {e}")

    # Prophet（和值/蓝球）
    try:
        pr_cfg = prophet_model.ProphetConfig()
        if getattr(args, "bayes_prophet", False):
            try:
                best_params, best_mae = prophet_model.bayes_optimize_prophet(
                    df, recent=args.bayes_prophet_recent, n_iter=args.bayes_prophet_calls
                )
                pr_cfg.changepoint_prior_scale = best_params["changepoint_prior_scale"]
                pr_cfg.seasonality_prior_scale = best_params["seasonality_prior_scale"]
                print(f"[bayes-prophet] 最优参数: {best_params}, mae={best_mae:.4f}")
            except Exception as e:
                print(f"[bayes-prophet][warn] 调参失败，回退默认: {e}")
        pr_resume = not getattr(args, "prophet_no_resume", False)
        pr_fresh = getattr(args, "prophet_fresh", False)
        pr_m, pr_cfg = prophet_model.train_prophet(df, pr_cfg, save_dir="models", resume=pr_resume, fresh=pr_fresh)

        def pred_pr(hist_df):
            out = prophet_model.predict_prophet(pr_m, hist_df)
            if out.get("red_probs"):
                out["red"] = {pos: out["red_probs"] for pos in range(1, 7)}
            if out.get("blue_probs"):
                out["blue"] = out["blue_probs"]
            return out

        base_predictors.append(pred_pr)
        print("[predict][prophet] added to blend")
    except Exception as e:
        print(f"[predict][prophet][warn] 跳过: {e}")

    if len(base_predictors) < 2:
        print("[predict][blend] 基础模型不足2个，跳过融合")
        return

    # 避免控制台乱码，使用 ASCII 文案
    print(f"[predict][blend] start blend with {len(base_predictors)} base models...")
    fused_blue_num, fused_blue_prob = blender.blend_blue_latest(df, base_predictors)
    fused_sum = blender.blend_sum_latest(df, base_predictors)
    fused_red = blender.blend_red_latest(df, base_predictors)
    print(f"[predict][blend] 融合蓝球 Top1: {fused_blue_num} (prob={fused_blue_prob:.3f})")
    print(f"[predict][blend] 融合和值预测≈{fused_sum:.2f}")
    print(f"[predict][blend] 融合红球预测: {fused_red}")

    # Stacking 元学习（蓝球 / 红球位置），使用概率向量特征
    try:
        # 强制启用贝叶斯调参的 stacking，确保 meta-learner 学习模型优劣
        n_calls = max(getattr(args, "stack_bayes_calls", 6), 10)
        stack_bayes = True
        # 蓝球 stacking
        scaler_b, meta_b, le_b = blender.train_stacking_blue(df, base_predictors, bayes=stack_bayes, n_iter=n_calls)
        stack_blue_num, stack_blue_prob = blender.predict_stacking_blue(df, base_predictors, scaler_b, meta_b, le_b)
        print(f"[predict][stack] 蓝球 Top1: {stack_blue_num} (prob={stack_blue_prob:.3f})")
        # 红球位置 stacking
        scalers_r, models_r, les_r = blender.train_stacking_red(df, base_predictors, bayes=stack_bayes, n_iter=n_calls)
        stack_red = blender.predict_stacking_red(df, base_predictors, scalers_r, models_r, les_r, topk=1)
        print(f"[predict][stack] 红球位置 Top1: {stack_red}")
    except Exception as e:
        print(f"[predict][stack][warn] 跳过 stacking: {e}")

    # 基于高概率红/蓝生成约束复式组合（和值范围由预测均值±预测标准差，奇偶默认3:3，若有奇偶预测则替换）
    try:
        fused_sum_int = int(round(fused_sum)) if fused_sum is not None else None
        sum_std = None
        try:
            sum_cfg = sum_model.SumStdConfig(window=args.window, iterations=150, depth=6, learning_rate=0.05)
            sum_m = sum_model.train_sumstd_model(df, sum_cfg, save_dir="models", resume=not args.cat_no_resume, fresh=False)
            sum_std = sum_model.predict_sumstd(sum_m, df, sum_cfg)
            print(f"[predict][sumstd] 预测和值标准差≈{sum_std:.2f}")
        except Exception as e:
            print(f"[predict][sumstd][warn] 跳过: {e}")
        if fused_sum_int is not None:
            pad = int(max(3, round(sum_std))) if sum_std is not None else 5
            sum_range = (fused_sum_int - pad, fused_sum_int + pad)
        else:
            sum_range = None
        odd_hint = None
        try:
            # 奇偶模型：从窗口中预测下一期红球奇数个数
            odd_cfg = odd_model.OddConfig(window=args.window, iterations=120, depth=6, learning_rate=0.05, topk=3)
            odd_m = odd_model.train_odd_model(df, odd_cfg, save_dir="models", resume=not args.cat_no_resume, fresh=False)
            odd_pred = odd_model.predict_odd(odd_m, df, odd_cfg)
            odd_hint = odd_pred["odd_pred"]
            print(f"[predict][odd] 奇数个数Top1={odd_pred['odd_pred']} 概率={odd_pred['odd_probs'][0][1]:.3f}")
        except Exception as e:
            print(f"[predict][odd][warn] 跳过: {e}")
        duplex = blender.generate_duplex_combos(
            df,
            base_predictors,
            top_red=10,
            top_blue=5,
            sum_range=sum_range,
            odd_count=odd_hint if odd_hint is not None else 3,
            max_combos=10,
        )
        if duplex:
            print("[predict][duplex] 推荐复式(前10):")
            for idx, c in enumerate(duplex, 1):
                reds = " ".join(f"{n:02d}" for n in c["reds"])
                print(f"  #{idx}: 红[{reds}] 蓝[{c['blue']:02d}] sum={c['sum']} odd={c['odd']} score={c['score']:.3f}")
    except Exception as e:
        print(f"[predict][duplex][warn] 生成失败: {e}")

    # 杀号：概率极低(<0.01%)的红/蓝
    try:
        kill = blender.kill_numbers(df, base_predictors, red_thresh=1e-4, blue_thresh=1e-4)
        if kill["red_kill"] or kill["blue_kill"]:
            print(f"[predict][kill] 红可剔除: {kill['red_kill']}")
            print(f"[predict][kill] 蓝可剔除: {kill['blue_kill']}")
    except Exception as e:
        print(f"[predict][kill][warn] 计算失败: {e}")


def cmd_predict_seq(args) -> None:
    db_path = Path(args.db)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if df.empty or len(df) < args.window + 1:
        print("样本不足，无法训练序列模型。")
        sys.exit(1)
    # 可选贝叶斯调参
    if getattr(args, "bayes_seq", False):
        try:
            best_params, best_loss = seq_model.bayes_optimize_seq(
                df, recent=max(200, args.recent or 0), n_iter=args.bayes_seq_calls
            )
            args.window = int(best_params["window"])
            args.batch = int(best_params["batch_size"])
            args.lr = float(best_params["lr"])
            args.d_model = int(best_params["d_model"])
            args.nhead = int(best_params["nhead"])
            args.layers = int(best_params["num_layers"])
            args.ff = int(best_params["ff"])
            args.dropout = float(best_params["dropout"])
            print(f"[bayes-seq] 最优参数: {best_params}, loss={best_loss:.4f}")
        except Exception as e:
            print(f"[bayes-seq][warn] 调参失败，回退命令行参数: {e}")
    cfg = seq_model.TrainConfig(
        window=args.window,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        ff=args.ff,
        dropout=args.dropout,
        topk=args.topk,
    )
    seq_resume = not getattr(args, "seq_no_resume", False)
    seq_fresh = getattr(args, "seq_fresh", False)
    model, cfg = seq_model.train_seq_model(df, cfg, save_dir="models", resume=seq_resume, fresh=seq_fresh)
    preds = seq_model.predict_seq(model, cfg, df)
    print(f"[predict-seq] 使用最近 {len(df)} 期训练，窗口 {cfg.window}，epochs {cfg.epochs}")
    for pos, items in preds["red"].items():
        print(f"[predict-seq] 位置{pos} Top{cfg.topk}: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
    print(f"[predict-seq] 蓝球 Top{cfg.topk}: " + ", ".join([f"{n}({p:.3f})" for n, p in preds['blue']]))


def cmd_predict_tft(args) -> None:
    db_path = Path(args.db)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if df.empty or len(df) < args.window + 1:
        print("样本不足，无法训练 TFT 序列模型。")
        sys.exit(1)
    cfg = tft_model.TFTConfig(
        window=args.window,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
        ff=args.ff,
        dropout=args.dropout,
        topk=args.topk,
    )
    tft_resume = not getattr(args, "tft_no_resume", False)
    tft_fresh = getattr(args, "tft_fresh", False)
    model, cfg = tft_model.train_tft(df, cfg, save_dir="models", resume=tft_resume, fresh=tft_fresh)
    preds = tft_model.predict_tft(model, cfg, df)
    print(f"[predict-tft] 使用最近 {len(df)} 期训练，窗口 {cfg.window}，epochs {cfg.epochs}")
    for pos, items in preds["red"].items():
        print(f"[predict-tft] 位置{pos} Top{cfg.topk}: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
    print(f"[predict-tft] 蓝球 Top{cfg.topk}: " + ", ".join([f"{n}({p:.3f})" for n, p in preds['blue']]))


def cmd_tune_cat(args) -> None:
    db_path = Path(args.db)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if df.empty or len(df) < args.window + 1:
        print("样本不足，无法进行贝叶斯调参。")
        sys.exit(1)
    print(f"[tune-cat] 使用最近 {len(df)} 期数据，窗口 {args.window}，迭代 {args.n_iter} 次，cv={args.cv}")
    try:
        best_model, best_params, best_score = ml_model.bayes_optimize_catboost(
            df,
            window=args.window,
            n_iter=args.n_iter,
            cv_splits=args.cv,
            save_dir="models",
        )
    except Exception as e:
        print(f"[tune-cat][error] {e}")
        sys.exit(1)
    print(f"[tune-cat] 最优参数: {best_params}")
    print(f"[tune-cat] 最优评分(neg_log_loss): {best_score:.4f}")
    preds = ml_model.predict_next({"red": {}, "blue": best_model, "window": args.window}, df, top_k=args.topk)
    print(f"[tune-cat] 蓝球 Top{args.topk}: " + ", ".join([f"{n}({p:.3f})" for n, p in preds['blue']]))




def cmd_cv_cat(args) -> None:
    db_path = Path(args.db)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if len(df) < args.train + args.test:
        print("样本不足，无法滚动验证。")
        sys.exit(1)

    def trainer(train_df: pd.DataFrame):
        return ml_model.train_models(
            train_df,
            window=args.window,
            iterations=args.iter,
            depth=args.depth,
            learning_rate=args.lr,
        )

    def predictor(model, hist_df: pd.DataFrame):
        return ml_model.predict_next(model, hist_df, top_k=1)

    results = validation.rolling_cv(
        df,
        trainer=trainer,
        predictor=predictor,
        train_size=args.train,
        test_size=args.test,
        step=args.step,
    )
    if not results:
        print("未得到有效折次。")
        return
    red_avg = sum(r.red_hit for r in results) / len(results)
    blue_avg = sum(r.blue_hit for r in results) / len(results)
    print(f"[cv-cat] 折数 {len(results)}, 红位置Top1均值 {red_avg:.3f}, 蓝Top1均值 {blue_avg:.3f}")
    for r in results:
        print(f"[cv-cat] fold{r.fold}: red={r.red_hit:.3f}, blue={r.blue_hit:.3f}, train_end={r.train_end}, test_end={r.test_end}")


def cmd_cv_tft(args) -> None:
    db_path = Path(args.db)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if len(df) < args.train + args.test:
        print("样本不足，无法滚动验证。")
        sys.exit(1)

    def trainer(train_df: pd.DataFrame):
        cfg = tft_model.TFTConfig(
            window=args.window,
            batch=args.batch,
            epochs=args.epochs,
            lr=args.lr,
            d_model=args.d_model,
            nhead=args.nhead,
            layers=args.layers,
            ff=args.ff,
            dropout=args.dropout,
            topk=1,
            freq_window=args.freq_window,
            entropy_window=args.entropy_window,
        )
        model, _ = tft_model.train_tft(train_df, cfg)
        return (model, cfg)

    def predictor(model_cfg, hist_df: pd.DataFrame):
        model, cfg = model_cfg
        return tft_model.predict_tft(model, cfg, hist_df)

    results = validation.rolling_cv_generic(
        df,
        trainer=trainer,
        predictor=predictor,
        train_size=args.train,
        test_size=args.test,
        step=args.step,
    )
    if not results:
        print("未得到有效折次。")
        return
    red_avg = sum(r.red_hit for r in results) / len(results)
    blue_avg = sum(r.blue_hit for r in results) / len(results)
    print(f"[cv-tft] 折数 {len(results)}, 红位置Top1均值 {red_avg:.3f}, 蓝Top1均值 {blue_avg:.3f}")
    for r in results:
        print(f"[cv-tft] fold{r.fold}: red={r.red_hit:.3f}, blue={r.blue_hit:.3f}, train_end={r.train_end}, test_end={r.test_end}")


def _sync_if_needed(db_path: Path, do_sync: bool) -> None:
    if not do_sync:
        return
    print("[train-all] 同步数据中...")
    database.init_db(db_path)
    all_draws = scraper.fetch_all_draws()
    with database.get_conn(db_path) as conn:
        existing = {row["issue"] for row in conn.execute("SELECT issue FROM draws")}
        to_insert = scraper.filter_new_draws(all_draws, existing)
        if not existing:
            to_insert = all_draws
        if to_insert:
            affected = database.upsert_draws(conn, to_insert)
            latest_issue = max(d.issue for d in to_insert)
            print(f"[train-all] 同步完成，新入库 {affected} 期，最新期号 {latest_issue}")
        else:
            print("[train-all] 数据库已最新，无需更新")


def cmd_train_all(args, cfg=None) -> None:
    db_path = Path(args.db)
    _sync_if_needed(db_path, args.sync)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if df.empty:
        print("数据库为空，请先同步数据。")
        sys.exit(1)

    print(f"[train-all] 使用最近 {len(df)} 期样本")

    # CatBoost
    if not args.no_cat:
        if len(df) < args.cat_window + 1:
            print("[train-all][CatBoost] 样本不足，跳过")
        elif args.pbt_cat:
            print("[train-all][CatBoost] PBT 演化训练中...")
            try:
                from lottery.ml_model import CatModelAdapter
                adapter = CatModelAdapter()
                if args.bayes_cat:
                    print("[train-all][CatBoost] 启动贝叶斯 PBT 初始化...")
                    try:
                        _, best_params, best_score = ml_model.bayes_optimize_catboost(
                            df, window=args.cat_window, n_iter=args.bayes_cat_calls, cv_splits=args.bayes_cat_cv, save_dir="models"
                        )
                        # Use best params to seed population with slight variation
                        initial_configs = []
                        base_depth = int(best_params.get("depth", 6))
                        base_lr = float(best_params.get("learning_rate", 0.1))
                        for _ in range(4):
                            initial_configs.append({
                                "window": args.cat_window,
                                "depth": base_depth + random.choice([-1, 0, 1]) if base_depth > 1 else max(1, base_depth + random.choice([0, 1])),
                                "learning_rate": base_lr * random.uniform(0.8, 1.2),
                            })
                        print(f"[train-all][CatBoost] 贝叶斯初始化完成: depth~{base_depth}, lr~{base_lr:.4f}")
                    except Exception as e:
                         print(f"[train-all][CatBoost][Bayes-Init] 失败，回退到随机: {e}")
                         initial_configs = [
                            {
                                "window": args.cat_window,
                                "depth": random.choice([4, 6, 8]),
                                "learning_rate": random.choice([0.05, 0.1, 0.2]),
                            }
                            for _ in range(4)
                        ]
                else:
                    initial_configs = [
                        {
                            "window": args.cat_window,
                            "depth": random.choice([4, 6, 8]),
                            "learning_rate": random.choice([0.05, 0.1, 0.2]),
                        }
                        for _ in range(4)
                    ]
                runner = pbt.PBTRunner(adapter, df, population_size=4, generations=args.pbt_generations, steps_per_gen=max(50, args.pbt_steps), save_dir="models/pbt_cat")
                runner.initialize(initial_configs)
                best_member = runner.run()
                models = best_member.model_state
                print(f"[train-all][CatBoost] PBT 完成，最佳得分: {best_member.performance:.4f}")
                
                # Predict
                preds = ml_model.predict_next(models, df, top_k=3)
                print("[train-all][CatBoost] 训练完成(PBT)，最新窗口预测：")
                for pos, items in preds["red"].items():
                    print(f"  位置{pos} Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
                print("  蓝球 Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in preds["blue"]]))
            except Exception as e:
                 print(f"[train-all][CatBoost][PBT] 失败: {e}")
        else:
            print("[train-all][CatBoost] 训练中...")
            cat_iter = args.cat_iter
            cat_depth = args.cat_depth
            cat_lr = args.cat_lr
            if args.bayes_cat:
                try:
                    _, best_params, best_score = ml_model.bayes_optimize_catboost(
                        df, window=args.cat_window, n_iter=args.bayes_cat_calls, cv_splits=args.bayes_cat_cv, save_dir="models"
                    )
                    cat_depth = int(best_params.get("depth", cat_depth))
                    cat_lr = float(best_params.get("learning_rate", cat_lr))
                    cat_iter = int(best_params.get("iterations", cat_iter))
                    print(f"[train-all][bayes-cat] 最优: depth={cat_depth}, lr={cat_lr:.4f}, iter={cat_iter}, score={best_score:.4f}")
                except Exception as e:
                    print(f"[train-all][bayes-cat][warn] 调参失败，沿用手动参数: {e}")
            cat_resume = not args.cat_no_resume
            cat_fresh = args.cat_fresh or args.fresh
            models = ml_model.train_models(
                df,
                window=args.cat_window,
                iterations=cat_iter,
                depth=cat_depth,
                learning_rate=cat_lr,
                save_dir="models",
                resume=cat_resume,
                fresh=cat_fresh,
            )
            preds = ml_model.predict_next(models, df, top_k=3)
            print("[train-all][CatBoost] 训练完成，最新窗口预测：")
            for pos, items in preds["red"].items():
                print(f"  位置{pos} Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
            print("  蓝球 Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in preds["blue"]]))

    # Transformer 序列
    if not args.no_seq:
        if len(df) < args.seq_window + 1:
            print("[train-all][Transformer] 样本不足，跳过")
        elif args.pbt_seq:
            print("[train-all][Transformer] PBT 演化训练中...")
            try:
                from lottery.seq_model import SeqModelAdapter, TrainConfig
                adapter = SeqModelAdapter()
                if args.bayes_seq:
                    print("[train-all][Transformer] 启动贝叶斯 PBT 初始化...")
                    try:
                        bp, _ = seq_model.bayes_optimize_seq(df, recent=args.bayes_seq_recent, n_iter=args.bayes_seq_calls)
                        initial_configs = []
                        for _ in range(4):
                            initial_configs.append(TrainConfig(
                                window=int(bp["window"]),
                                batch_size=int(bp["batch_size"]),
                                epochs=1,
                                lr=float(bp["lr"]) * random.uniform(0.8, 1.2),
                                d_model=int(bp["d_model"]),
                                nhead=int(bp["nhead"]),
                                num_layers=int(bp["num_layers"]),
                                ff=int(bp["ff"]),
                                dropout=float(bp["dropout"])
                            ))
                        print(f"[train-all][Transformer] 贝叶斯初始化完成")
                    except Exception as e:
                        print(f"[train-all][Transformer][Bayes-Init] 失败，回退随机: {e}")
                        initial_configs = [
                            TrainConfig(
                                window=args.seq_window,
                                batch_size=64,
                                epochs=1,
                                lr=random.choice([1e-3, 5e-4]),
                                d_model=96,
                                nhead=4,
                                num_layers=3,
                                ff=192,
                                dropout=0.1
                            )
                            for _ in range(4)
                        ]
                else:
                    initial_configs = [
                        TrainConfig(
                            window=args.seq_window,
                            batch_size=64,
                            epochs=1,
                            lr=random.choice([1e-3, 5e-4]),
                            d_model=96,
                            nhead=4,
                            num_layers=3,
                            ff=192,
                            dropout=0.1
                        )
                        for _ in range(4)
                    ]
                runner = pbt.PBTRunner(adapter, df, population_size=4, generations=args.pbt_generations, steps_per_gen=args.pbt_epochs, save_dir="models/pbt_seq")
                runner.initialize(initial_configs)
                best_member = runner.run()
                model = best_member.model_state
                cfg = best_member.config
                preds = seq_model.predict_seq(model, cfg, df)
                print("[train-all][Transformer] PBT 完成，最新窗口预测：")
                for pos, items in preds["red"].items():
                    print(f"  位置{pos} Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
                print("  蓝球 Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in preds['blue']]))
            except Exception as e:
                print(f"[train-all][Transformer][PBT] 失败: {e}")
        else:
            print("[train-all][Transformer] 训练中...")
            seq_window = args.seq_window
            seq_bs = 64
            seq_epochs = args.seq_epochs
            seq_lr = args.seq_lr
            seq_dm = args.seq_d_model
            seq_nh = args.seq_nhead
            seq_layers = args.seq_layers
            seq_ff = args.seq_ff
            seq_dr = args.seq_dropout
            if args.bayes_seq:
                try:
                    best_params, best_loss = seq_model.bayes_optimize_seq(
                        df, recent=args.bayes_seq_recent, n_iter=args.bayes_seq_calls
                    )
                    seq_window = int(best_params["window"])
                    seq_bs = int(best_params["batch_size"])
                    seq_lr = float(best_params["lr"])
                    seq_dm = int(best_params["d_model"])
                    seq_nh = int(best_params["nhead"])
                    seq_layers = int(best_params["num_layers"])
                    seq_ff = int(best_params["ff"])
                    seq_dr = float(best_params["dropout"])
                    print(f"[train-all][bayes-seq] 最优: {best_params}, loss={best_loss:.4f}")
                except Exception as e:
                    print(f"[train-all][bayes-seq][warn] 调参失败，沿用手动参数: {e}")
            cfg = seq_model.TrainConfig(
                window=seq_window,
                batch_size=seq_bs,
                epochs=seq_epochs,
                lr=seq_lr,
                d_model=seq_dm,
                nhead=seq_nh,
                num_layers=seq_layers,
                ff=seq_ff,
                dropout=seq_dr,
                topk=3,
            )
            seq_resume = not args.seq_no_resume
            seq_fresh = args.seq_fresh or args.fresh
            model, cfg = seq_model.train_seq_model(df, cfg, save_dir="models", resume=seq_resume, fresh=seq_fresh)
            preds = seq_model.predict_seq(model, cfg, df)
            print("[train-all][Transformer] 训练完成，最新窗口预测：")
            for pos, items in preds["red"].items():
                print(f"  位置{pos} Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
            print("  蓝球 Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in preds['blue']]))

    # TFT
    if args.run_tft:
        if len(df) < args.tft_window + 1:
            print("[train-all][TFT] 样本不足，跳过")
        elif args.pbt_tft:
            print("[train-all][TFT] PBT 演化训练中...")
            try:
                from lottery.tft_model import TftModelAdapter, TFTConfig
                adapter = TftModelAdapter()
                # Check for Bayes/Random Tuning to init
                if getattr(args, "bayes_tft", False):
                     print("[train-all][TFT] 启动贝叶斯 PBT 初始化...")
                     try:
                         best_params, best_loss = tft_model.bayes_optimize_tft(
                             df, recent=getattr(args, "bayes_tft_recent", 400), n_iter=getattr(args, "bayes_tft_calls", 10)
                         )
                         initial_configs = []
                         base_lr = float(best_params["lr"])
                         base_dr = float(best_params["dropout"])
                         for _ in range(4):
                             # Perturb slightly
                             initial_configs.append(TFTConfig(
                                 window=int(best_params["window"]),
                                 batch=int(best_params["batch"]),
                                 epochs=1,
                                 lr=base_lr * random.uniform(0.8, 1.2),
                                 d_model=int(best_params["d_model"]),
                                 nhead=int(best_params["nhead"]),
                                 layers=int(best_params["layers"]),
                                 ff=int(best_params["ff"]),
                                 dropout=base_dr,
                             ))
                         print(f"[train-all][TFT] 贝叶斯初始化完成: {best_params}")
                     except Exception as e:
                         print(f"[train-all][TFT][Bayes-Init] 失败: {e}")
                         # Fallback to random
                         initial_configs = []

                if not initial_configs:
                     initial_configs = [
                        TFTConfig(
                             window=args.tft_window,
                             batch=64,
                             epochs=1,
                             lr=random.choice([1e-3, 5e-4]),
                             d_model=128,
                             nhead=4,
                             layers=3,
                        )
                        for _ in range(4)
                     ]
                runner = pbt.PBTRunner(adapter, df, population_size=4, generations=args.pbt_generations, steps_per_gen=args.pbt_epochs, save_dir="models/pbt_tft")
                runner.initialize(initial_configs)
                best_member = runner.run()
                model = best_member.model_state
                cfg = best_member.config
                preds = tft_model.predict_tft(model, cfg, df)
                print("[train-all][TFT] PBT 完成，最新窗口预测：")
                for pos, items in preds["red"].items():
                    print(f"  位置{pos} Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
                print("  蓝球 Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in preds['blue']]))
            except Exception as e:
                 print(f"[train-all][TFT][PBT] 失败: {e}")
        else:
            print("[train-all][TFT] 训练中...（若无 GPU 将较耗时）")
            cfg = tft_model.TFTConfig(
                window=args.tft_window,
                batch=args.tft_batch,
                epochs=args.tft_epochs,
                lr=args.tft_lr,
                d_model=args.tft_d_model,
                nhead=args.tft_nhead,
                layers=args.tft_layers,
                ff=args.tft_ff,
                dropout=args.tft_dropout,
                topk=3,
                freq_window=args.tft_freq_window,
                entropy_window=args.tft_entropy_window,
            )
            tft_resume = not args.tft_no_resume
            tft_fresh = args.tft_fresh or args.fresh
            model, cfg = tft_model.train_tft(df, cfg, save_dir="models", resume=tft_resume, fresh=tft_fresh)
            preds = tft_model.predict_tft(model, cfg, df)
            print("[train-all][TFT] 训练完成，最新窗口预测：")
            for pos, items in preds["red"].items():
                print(f"  位置{pos} Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in items]))
            print("  蓝球 Top3: " + ", ".join([f"{n}({p:.3f})" for n, p in preds['blue']]))

    # N-HiTS（和值+蓝球单变量通道）
    if args.run_nhits:
        if len(df) < args.nhits_input + 1:
            print("[train-all][N-HiTS] 样本不足，跳过")
        elif args.pbt_nhits:
             print("[train-all][N-HiTS] PBT 演化训练中...")
             try:
                 from lottery.nhits_model import NHitsModelAdapter, NHitsConfig
                 adapter = NHitsModelAdapter()
                 if args.bayes_nhits:
                      print("[train-all][N-HiTS] 启动贝叶斯 PBT 初始化...")
                      try:
                          best_params, best_loss = nhits_model.bayes_optimize_nhits(
                              df, recent=getattr(args, "bayes_nhits_recent", 300), n_iter=getattr(args, "bayes_nhits_calls", 10)
                          )
                          initial_configs = []
                          base_lr = float(best_params["learning_rate"])
                          for _ in range(4):
                              initial_configs.append(NHitsConfig(
                                  input_size=int(best_params["input_size"]),
                                  learning_rate=base_lr * random.uniform(0.8, 1.2),
                                  max_steps=int(best_params["max_steps"]),
                                  batch_size=int(best_params["batch_size"]),
                                  valid_size=0.1
                              ))
                          print(f"[train-all][N-HiTS] 贝叶斯初始化完成: {best_params}")
                      except Exception as e:
                          print(f"[train-all][N-HiTS][Bayes-Init] 失败: {e}")
                          initial_configs = []

                 if not initial_configs:
                     initial_configs = [
                         NHitsConfig(
                             input_size=args.nhits_input,
                             learning_rate=random.choice([1e-3, 5e-4]),
                             max_steps=args.nhits_steps,
                             batch_size=32,
                             valid_size=0.1
                         )
                         for _ in range(4)
                     ]
                 runner = pbt.PBTRunner(adapter, df, population_size=4, generations=args.pbt_generations, steps_per_gen=args.pbt_steps, save_dir="models/pbt_nhits")
                 runner.initialize(initial_configs)
                 best_member = runner.run()
                 nf = best_member.model_state
                 cfg = best_member.config
                 preds = nhits_model.predict_nhits(nf, cfg, df)
                 print(f"[train-all][N-HiTS] PBT 完成，和值预测≈{preds['sum_pred']:.2f}，蓝球预测≈{preds['blue_pred']:.2f}")
             except Exception as e:
                 print(f"[train-all][N-HiTS][PBT] 失败: {e}")
        else:
            print("[train-all][N-HiTS] 训练中...")
            cfg = nhits_model.NHitsConfig(
                input_size=args.nhits_input,
                h=1,
                n_layers=args.nhits_layers,
                n_blocks=args.nhits_blocks,
                n_harmonics=2,
                n_polynomials=1,
                dropout=0.1,
                learning_rate=args.nhits_lr,
                max_steps=args.nhits_steps,
                batch_size=32,
                valid_size=0.1,
                topk=3,
            )
            nh_resume = not getattr(args, "nhits_no_resume", False)
            nh_fresh = getattr(args, "nhits_fresh", False) or args.fresh
            nf, cfg = nhits_model.train_nhits(df, cfg, save_dir="models", resume=nh_resume, fresh=nh_fresh)
            preds = nhits_model.predict_nhits(nf, cfg, df)
            print(f"[train-all][N-HiTS] 训练完成，和值预测≈{preds['sum_pred']:.2f}，蓝球预测≈{preds['blue_pred']:.2f}")

    # Prophet（和值/蓝球单变量）
    if args.run_prophet:
        if len(df) < 20:
            print("[train-all][Prophet] 样本不足，跳过")
        elif args.pbt_prophet:
            print("[train-all][Prophet] PBT 演化训练中...")
            try:
                from lottery.prophet_model import ProphetModelAdapter, ProphetConfig
                adapter = ProphetModelAdapter()
                if args.bayes_prophet:
                    print("[train-all][Prophet] 启动贝叶斯 PBT 初始化...")
                    try:
                        best_params, best_score = prophet_model.bayes_optimize_prophet(
                            df, recent=getattr(args, "bayes_prophet_recent", 200), n_iter=getattr(args, "bayes_prophet_calls", 10)
                        )
                        initial_configs = []
                        base_cps = float(best_params["changepoint_prior_scale"])
                        base_sps = float(best_params["seasonality_prior_scale"])
                        for _ in range(4):
                            initial_configs.append(ProphetConfig(
                                changepoint_prior_scale=base_cps * random.uniform(0.8, 1.2),
                                seasonality_prior_scale=base_sps * random.uniform(0.8, 1.2)
                            ))
                        print(f"[train-all][Prophet] 贝叶斯初始化完成: {best_params}")
                    except Exception as e:
                        print(f"[train-all][Prophet][Bayes-Init] 失败: {e}")
                        initial_configs = []

                if not initial_configs:
                    initial_configs = [
                        ProphetConfig(
                            changepoint_prior_scale=random.choice([0.01, 0.05, 0.1]),
                            seasonality_prior_scale=random.choice([1.0, 10.0])
                        )
                        for _ in range(4)
                    ]
                runner = pbt.PBTRunner(adapter, df, population_size=4, generations=args.pbt_generations, steps_per_gen=args.pbt_epochs, save_dir="models/pbt_prophet")
                runner.initialize(initial_configs)
                best_member = runner.run()
                models = best_member.model_state
                preds = prophet_model.predict_prophet(models, df)
                print(f"[train-all][Prophet] PBT 完成，和值预测≈{preds['sum']:.2f}，蓝球预测≈{preds['blue']:.2f}")
            except Exception as e:
                print(f"[train-all][Prophet][PBT] 失败: {e}")
        else:
            print("[train-all][Prophet] 训练中...")
            cfg = prophet_model.ProphetConfig()
            pr_resume = not getattr(args, "prophet_no_resume", False)
            pr_fresh = getattr(args, "prophet_fresh", False) or args.fresh
            models, cfg = prophet_model.train_prophet(df, cfg, save_dir="models", resume=pr_resume, fresh=pr_fresh)
            preds = prophet_model.predict_prophet(models, df)
            print(f"[train-all][Prophet] 训练完成，和值预测≈{preds['sum']:.2f}，蓝球预测≈{preds['blue']:.2f}")

    # TimesNet（和值+蓝球单变量通道，基于 NeuralForecast）
    if args.run_timesnet:
        if len(df) < args.timesnet_input + 1:
            print("[train-all][TimesNet] 样本不足，跳过")
        elif args.pbt_timesnet:
            print("[train-all][TimesNet] PBT 演化训练中...")
            try:
                from lottery.timesnet_model import TimesNetModelAdapter, TimesNetConfig
                adapter = TimesNetModelAdapter()
                if args.bayes_timesnet:
                    print("[train-all][TimesNet] 启动贝叶斯 PBT 初始化...")
                    try:
                        best_params, best_loss = timesnet_model.bayes_optimize_timesnet(
                            df, recent=getattr(args, "bayes_timesnet_recent", 400), n_iter=getattr(args, "bayes_timesnet_calls", 10)
                        )
                        initial_configs = []
                        base_lr = float(best_params["learning_rate"])
                        base_dr = float(best_params["dropout"])
                        for _ in range(4):
                            initial_configs.append(TimesNetConfig(
                                input_size=int(best_params["input_size"]),
                                hidden_size=int(best_params["hidden_size"]),
                                top_k=int(best_params["top_k"]),
                                learning_rate=base_lr * random.uniform(0.8, 1.2),
                                dropout=base_dr,
                                max_steps=int(best_params["max_steps"]),
                                batch_size=int(best_params["batch_size"]),
                                valid_size=0.1
                            ))
                        print(f"[train-all][TimesNet] 贝叶斯初始化完成: {best_params}")
                    except Exception as e:
                        print(f"[train-all][TimesNet][Bayes-Init] 失败: {e}")
                        initial_configs = []

                if not initial_configs:
                    initial_configs = [
                        TimesNetConfig(
                            input_size=args.timesnet_input,
                            hidden_size=args.timesnet_hidden,
                            learning_rate=random.choice([1e-3, 5e-4]),
                            max_steps=args.timesnet_steps,
                            batch_size=32,
                            valid_size=0.1
                        )
                        for _ in range(4)
                    ]
                runner = pbt.PBTRunner(adapter, df, population_size=4, generations=args.pbt_generations, steps_per_gen=args.pbt_steps, save_dir="models/pbt_timesnet")
                runner.initialize(initial_configs)
                best_member = runner.run()
                nf = best_member.model_state
                preds = timesnet_model.predict_timesnet(nf, df)
                print(f"[train-all][TimesNet] PBT 完成，和值预测≈{preds['sum_pred']:.2f}，蓝球预测≈{preds['blue_pred']:.2f}")
            except Exception as e:
                print(f"[train-all][TimesNet][PBT] 失败: {e}")
        else:
            print("[train-all][TimesNet] 训练中...")
            cfg = timesnet_model.TimesNetConfig(
                input_size=args.timesnet_input,
                h=1,
                hidden_size=args.timesnet_hidden,
                top_k=args.timesnet_topk,
                dropout=args.timesnet_dropout,
                learning_rate=args.timesnet_lr,
                max_steps=args.timesnet_steps,
                batch_size=32,
                valid_size=0.1,
            )
            tn_resume = not getattr(args, "timesnet_no_resume", False)
            tn_fresh = getattr(args, "timesnet_fresh", False) or args.fresh
            nf, cfg = timesnet_model.train_timesnet(df, cfg, save_dir="models", resume=tn_resume, fresh=tn_fresh)
            preds = timesnet_model.predict_timesnet(nf, df)
            print(f"[train-all][TimesNet] 训练完成，和值预测≈{preds['sum_pred']:.2f}，蓝球预测≈{preds['blue_pred']:.2f}")

    # Blender（蓝球/和值/红球位置动态加权示例）
    if args.run_blend:
        print("[train-all][Blender] 构建基础预测并训练融合模型...")

        base_predictors = []

        # 0) 频率与随机基线（Random Fallback / Baseline）
        # 频率基线
        def pred_freq(hist_df):
            cols = ["red1", "red2", "red3", "red4", "red5", "red6", "blue"]
            data = hist_df[cols].to_numpy(dtype=int)
            reds = data[:, :6].ravel()
            blues = data[:, 6]
            red_freq = np.bincount(reds, minlength=34)
            blue_freq = np.bincount(blues, minlength=17)
            red_top = np.argsort(red_freq[1:])[::-1][:6] + 1
            blue_top = np.argsort(blue_freq[1:])[::-1][:1] + 1
            red_preds = {i + 1: [(int(n), 1.0 / len(red_top))] for i, n in enumerate(red_top[:6])}
            blue_preds = [(int(blue_top[0]), 1.0)]
            return {"red": red_preds, "blue": blue_preds, "sum_pred": float(data[-1, :].sum())}
        base_predictors.append(pred_freq)

        # 随机基线（兜底）
        def pred_random(hist_df):
            rng = np.random.default_rng(len(hist_df))
            red_nums = rng.choice(np.arange(1, 34), size=6, replace=False)
            blue_num = int(rng.integers(1, 17))
            red_preds = {i + 1: [(int(n), 1.0 / 6)] for i, n in enumerate(red_nums)}
            blue_preds = [(blue_num, 1.0)]
            last_sum = float(hist_df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].iloc[-1].sum())
            return {"red": red_preds, "blue": blue_preds, "sum_pred": last_sum}
        base_predictors.append(pred_random)
        print(f"[train-all][Blender] 已添加频率与随机基线 predictors")
        # 1) CatBoost
        if not args.no_cat and len(df) >= args.cat_window + 1:
            cat_models = ml_model.train_models(
                df,
                window=args.cat_window,
                iterations=args.cat_iter,
                depth=args.cat_depth,
                learning_rate=args.cat_lr,
                resume=not args.cat_no_resume,
                fresh=args.cat_fresh or args.fresh
            )

            # Pre-compute all predictions for Blender (Speed optimization)
            print("[train-all][CatBoost] Batch predicting for Blender...")
            cat_batch_preds = ml_model.batch_predict(cat_models, df, top_k=1)
            
            def pred_cat(hist_df):
                # O(1) lookup using absolute index
                idx = hist_df.index[-1]
                if idx in cat_batch_preds:
                    return cat_batch_preds[idx]
                # Fallback for unexpected cases
                return ml_model.predict_next(cat_models, hist_df, top_k=1)

            base_predictors.append(pred_cat)

        # 2) Transformer
        if not args.no_seq and len(df) >= args.seq_window + 1:
            seq_cfg = seq_model.TrainConfig(
                window=args.seq_window,
                batch_size=64,
                epochs=max(1, min(args.seq_epochs, 10)),  # 融合阶段减小训练轮数加速
                lr=args.seq_lr,
                d_model=args.seq_d_model,
                nhead=args.seq_nhead,
                num_layers=args.seq_layers,
                ff=args.seq_ff,
                dropout=args.seq_dropout,
                topk=1,
            )
            seq_resume = not args.seq_no_resume
            seq_fresh = args.seq_fresh or args.fresh
            seq_m, seq_cfg = seq_model.train_seq_model(df, seq_cfg, save_dir="models", resume=seq_resume, fresh=seq_fresh)

            print("[train-all][Transformer] Batch predicting for Blender...")
            seq_batch_preds = seq_model.batch_predict_seq(seq_m, seq_cfg, df)
            
            def pred_seq(hist_df):
                idx = hist_df.index[-1]
                if idx in seq_batch_preds:
                    return seq_batch_preds[idx]
                return seq_model.predict_seq(seq_m, seq_cfg, hist_df)

            base_predictors.append(pred_seq)

        # 3) TFT
        if args.run_tft and len(df) >= args.tft_window + 1:
            tft_cfg = tft_model.TFTConfig(
                window=args.tft_window,
                batch=args.tft_batch,
                epochs=max(1, min(args.tft_epochs, 10)),
                lr=args.tft_lr,
                d_model=args.tft_d_model,
                nhead=args.tft_nhead,
                layers=args.tft_layers,
                ff=args.tft_ff,
                dropout=args.tft_dropout,
                topk=1,
                freq_window=args.tft_freq_window,
                entropy_window=args.tft_entropy_window,
            )
            tft_resume = not args.tft_no_resume
            tft_fresh = args.tft_fresh
            tft_m, tft_cfg = tft_model.train_tft(df, tft_cfg, save_dir="models", resume=tft_resume, fresh=tft_fresh)

            print("[train-all][TFT] Batch predicting for Blender...")
            tft_batch_preds = tft_model.batch_predict_tft(tft_m, tft_cfg, df)

            def pred_tft(hist_df):
                idx = hist_df.index[-1]
                if idx in tft_batch_preds:
                    return tft_batch_preds[idx]
                return tft_model.predict_tft(tft_m, tft_cfg, hist_df)

            base_predictors.append(pred_tft)

        # 4) N-HiTS（取蓝球点预测作为特征）
        if args.run_nhits and len(df) >= args.nhits_input + 1:
            nh_cfg = nhits_model.NHitsConfig(
                input_size=args.nhits_input,
                h=1,
                n_layers=args.nhits_layers,
                n_blocks=args.nhits_blocks,
                n_harmonics=2,
                n_polynomials=1,
                dropout=0.1,
                learning_rate=args.nhits_lr,
                max_steps=max(50, min(args.nhits_steps, 200)),
                batch_size=32,
                valid_size=0.1,
                topk=1,
            )
            nh_resume = not args.nhits_no_resume
            nh_fresh = args.nhits_fresh
            nh_m, nh_cfg = nhits_model.train_nhits(df, nh_cfg, save_dir="models", resume=nh_resume, fresh=nh_fresh)

            print("[train-all][N-HiTS] Batch predicting for Blender...")
            nh_batch_preds = nhits_model.batch_predict_nhits(nh_m, nh_cfg, df)

            def pred_nh(hist_df):
                idx = hist_df.index[-1]
                if idx in nh_batch_preds:
                    return nh_batch_preds[idx]
                return nhits_model.predict_nhits(nh_m, nh_cfg, hist_df)

            base_predictors.append(pred_nh)
            if getattr(args, "nhits_backtest", False):
                try:
                    bt = nhits_model.backtest_nhits_model(nh_m, nh_cfg, df, max_samples=args.nhits_backtest_samples)
                    print(f"[predict][nhits][backtest] blue_top1={bt['blue_top1']:.3f}, samples={bt['samples']}")
                except Exception as e:
                    print(f"[predict][nhits][backtest][warn] {e}")

        # 5) TimesNet（蓝球点预测）
        if args.run_timesnet and len(df) >= args.timesnet_input + 1:
            tn_cfg = timesnet_model.TimesNetConfig(
                input_size=args.timesnet_input,
                h=1,
                hidden_size=args.timesnet_hidden,
                top_k=args.timesnet_topk,
                dropout=args.timesnet_dropout,
                learning_rate=args.timesnet_lr,
                max_steps=max(50, min(args.timesnet_steps, 300)),
                batch_size=32,
                valid_size=0.1,
            )
            tn_resume = not args.timesnet_no_resume
            tn_fresh = args.timesnet_fresh
            tn_m, tn_cfg = timesnet_model.train_timesnet(df, tn_cfg, save_dir="models", resume=tn_resume, fresh=tn_fresh)

            print("[train-all][TimesNet] Batch predicting for Blender...")
            tn_batch_preds = timesnet_model.batch_predict_timesnet(tn_m, tn_cfg, df)

            def pred_tn(hist_df):
                idx = hist_df.index[-1]
                if idx in tn_batch_preds:
                    return tn_batch_preds[idx]
                return timesnet_model.predict_timesnet(tn_m, hist_df)

            base_predictors.append(pred_tn)
            if getattr(args, "timesnet_backtest", False):
                try:
                    bt = timesnet_model.backtest_timesnet_model(tn_m, tn_cfg, df, max_samples=args.timesnet_backtest_samples)
                    print(f"[predict][timesnet][backtest] blue_top1={bt['blue_top1']:.3f}, samples={bt['samples']}")
                except Exception as e:
                    print(f"[predict][timesnet][backtest][warn] {e}")

        # 6) Prophet（蓝球点预测）
        if args.run_prophet and len(df) >= 30:
            pr_cfg = prophet_model.ProphetConfig()
            pr_resume = not args.prophet_no_resume
            pr_fresh = args.prophet_fresh
            pr_m, pr_cfg = prophet_model.train_prophet(df, pr_cfg, save_dir="models", resume=pr_resume, fresh=pr_fresh)

            print("[train-all][Prophet] Batch predicting for Blender...")
            pr_batch_preds = prophet_model.batch_predict_prophet(pr_m, df)

            def pred_prophet(hist_df):
                idx = hist_df.index[-1]
                if idx in pr_batch_preds:
                    return pr_batch_preds[idx]
                return prophet_model.predict_prophet(pr_m, hist_df)

            base_predictors.append(pred_prophet)

        if len(base_predictors) < 2:
            print("[train-all][Blender] 基础模型不足2个，跳过融合")
        else:
            blend_cfg = blender.BlendConfig(train_size=args.blend_train, test_size=args.blend_test, step=args.blend_step, alpha=0.3, l1_ratio=0.1)
            # 蓝球
            avg_acc_b, folds_b = blender.rolling_blend_blue(df, base_predictors, cfg=blend_cfg)
            fused_num_b, fused_prob_b = blender.blend_blue_latest(df, base_predictors)
            print(f"[train-all][Blender] 蓝球 Top1 平均命中率: {avg_acc_b:.3f} (折数 {len(folds_b)})，最新融合预测: {fused_num_b} (prob={fused_prob_b:.3f})")
            # 和值
            avg_mae_s, folds_s = blender.rolling_blend_sum(df, base_predictors, cfg=blend_cfg)
            fused_sum = blender.blend_sum_latest(df, base_predictors)
            print(f"[train-all][Blender] 和值融合 MAE: {avg_mae_s:.3f} (折数 {len(folds_s)})，最新融合和值≈{fused_sum:.2f}")
            # 红球位置
            avg_acc_r, folds_r = blender.rolling_blend_red(df, base_predictors, cfg=blend_cfg)
            fused_red = blender.blend_red_latest(df, base_predictors)
            # Format to {2, 4, 20, 18, 15, 14} style
            fused_red_vals = list(fused_red.values())
            # Ensure sorted by key if needed, or just values in P1-P6 order:
            # blend_red_latest returns {pos: num}, pos is 1..6
            fused_red_list = [fused_red[pos] for pos in sorted(fused_red.keys())]
            fused_red_str = "{" + ", ".join(map(str, fused_red_list)) + "}"
            print(f"[train-all][Blender] 红球位置 Top1 平均命中率: {avg_acc_r:.3f} (折数 {len(folds_r)})，最新融合红球: {fused_red_str}")

    print("[train-all] 执行完毕")

def cmd_train_pbt(args) -> None:
    db_path = Path(args.db)
    with database.get_conn(db_path) as conn:
        df = analyzer.load_dataframe(conn, recent=args.recent)
    if df.empty:
        print("数据库为空或样本不足。")
        sys.exit(1)

    print(f"[PBT] Target Model: {args.model}, Population: {args.pop_size}, Generations: {args.generations}")

    dataset = df
    adapter = None
    initial_configs = []

    if args.model == "seq":
        from lottery.seq_model import SeqModelAdapter, TrainConfig
        adapter = SeqModelAdapter()
        # Create initial configs with some random variation or default
        base_cfg = TrainConfig(
            window=args.window,
            batch_size=64,
            epochs=1, # Not used in PBT step, but good for init
            lr=1e-3,
            d_model=96,
            nhead=4,
            num_layers=3,
            ff=192,
            dropout=0.1
        )
        initial_configs = [base_cfg for _ in range(args.pop_size)]
    elif args.model == "cat":
        from lottery.ml_model import CatModelAdapter
        adapter = CatModelAdapter()
        base_cfg = {
            "window": args.window,
            "depth": 6,
            "learning_rate": 0.1,
        }
        initial_configs = [
            {
                "window": args.window,
                "depth": random.choice([4, 6, 8]),
                "learning_rate": random.choice([0.05, 0.1, 0.2]),
            }
            for _ in range(args.pop_size)
        ]
    elif args.model == "tft":
        from lottery.tft_model import TftModelAdapter, TFTConfig
        adapter = TftModelAdapter()
        base_cfg = TFTConfig(
            window=args.window,
            batch=64,
            epochs=1,
            lr=1e-3,
            d_model=128,
            nhead=4,
            layers=3,
        )
        initial_configs = [base_cfg for _ in range(args.pop_size)]
    elif args.model == "nhits":
        from lottery.nhits_model import NHitsModelAdapter, NHitsConfig
        adapter = NHitsModelAdapter()
        base_cfg = NHitsConfig(
            input_size=60,
            learning_rate=1e-3,
            max_steps=100,
            batch_size=32,
            valid_size=0.1
        )
        initial_configs = [base_cfg for _ in range(args.pop_size)]
    elif args.model == "timesnet":
        from lottery.timesnet_model import TimesNetModelAdapter, TimesNetConfig
        adapter = TimesNetModelAdapter()
        base_cfg = TimesNetConfig(
            input_size=60,
            hidden_size=64,
            learning_rate=1e-3,
            max_steps=100,
            batch_size=32,
            valid_size=0.1
        )
        initial_configs = [base_cfg for _ in range(args.pop_size)]
    elif args.model == "prophet":
        from lottery.prophet_model import ProphetModelAdapter, ProphetConfig
        adapter = ProphetModelAdapter()
        base_cfg = ProphetConfig(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0
        )
        initial_configs = [base_cfg for _ in range(args.pop_size)]
    else:
        print(f"Model {args.model} not yet supported for PBT.")
        sys.exit(1)

    runner = pbt.PBTRunner(
        adapter=adapter,
        dataset=dataset,
        population_size=args.pop_size,
        generations=args.generations,
        steps_per_gen=args.steps_per_gen,
        fraction=0.2,
        save_dir=f"models/pbt_{args.model}",
        n_jobs=args.n_jobs
    )
    runner.initialize(initial_configs)
    runner.run()
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="双色球爬取与混沌概率分析")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="抓取/增量更新双色球开奖数据")
    p_sync.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_sync.set_defaults(func=cmd_sync)

    p_analyze = sub.add_parser("analyze", help="分析数据库中的开奖数据")
    p_analyze.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_analyze.add_argument("--recent", type=int, help="仅分析最近 N 期")
    p_analyze.add_argument(
        "--entropy-window",
        type=int,
        default=60,
        help="滑动窗口大小，用于熵与混沌指标计算（单位：号码个数）",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_pred = sub.add_parser("predict", help="使用 CatBoost 进行下一期概率预测（位置模型）")
    p_pred.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_pred.add_argument("--recent", type=int, default=400, help="训练使用的最近 N 期样本（默认400）")
    p_pred.add_argument("--window", type=int, default=10, help="滑窗长度")
    p_pred.add_argument("--topk", type=int, default=3, help="每个位置输出前 topk 个号码及概率")
    p_pred.add_argument("--iter", type=int, default=300, help="CatBoost 迭代轮数")
    p_pred.add_argument("--depth", type=int, default=6, help="CatBoost 树深度")
    p_pred.add_argument("--lr", type=float, default=0.1, help="CatBoost 学习率")
    p_pred.add_argument("--cat-fresh", action="store_true", help="强制重训 CatBoost 并覆盖保存")
    p_pred.add_argument("--cat-no-resume", action="store_true", help="不加载已保存 CatBoost 模型")
    p_pred.add_argument("--bayes-cat", action="store_true", help="启用 CatBoost 贝叶斯调参")
    p_pred.add_argument("--bayes-cat-calls", type=int, default=8, help="CatBoost 贝叶斯搜索迭代次数")
    p_pred.add_argument("--bayes-cat-cv", type=int, default=3, help="CatBoost 贝叶斯调参交叉验证折数")
    p_pred.add_argument("--seq-fresh", action="store_true", help="强制重训 Transformer 并覆盖保存")
    p_pred.add_argument("--seq-no-resume", action="store_true", help="不加载已保存 Transformer 模型")
    p_pred.add_argument("--tft-fresh", action="store_true", help="强制重训 TFT 并覆盖保存")
    p_pred.add_argument("--tft-no-resume", action="store_true", help="不加载已保存 TFT 模型")
    p_pred.add_argument("--bayes-seq", action="store_true", help="启用 Transformer 贝叶斯调参")
    p_pred.add_argument("--bayes-seq-calls", type=int, default=6, help="Transformer 贝叶斯搜索迭代次数")
    p_pred.add_argument("--bayes-seq-recent", type=int, default=300, help="Transformer 贝叶斯调参使用的最近样本数")
    p_pred.add_argument("--seq-backtest", action="store_true", help="对 Transformer 进行流式回测")
    p_pred.add_argument("--seq-backtest-batch", type=int, default=128, help="回测批大小")
    p_pred.add_argument("--bayes-tft", action="store_true", help="启用 TFT 贝叶斯调参")
    p_pred.add_argument("--bayes-tft-calls", type=int, default=4, help="TFT 贝叶斯搜索迭代次数")
    p_pred.add_argument("--bayes-tft-recent", type=int, default=400, help="TFT 贝叶斯调参使用的最近样本数")
    p_pred.add_argument("--tft-rand", action="store_true", help="启用 TFT 随机调参兜底")
    p_pred.add_argument("--tft-rand-samples", type=int, default=80, help="TFT 随机搜索采样数")
    p_pred.add_argument("--tft-backtest", action="store_true", help="对 TFT 进行流式回测")
    p_pred.add_argument("--tft-backtest-batch", type=int, default=128, help="TFT 回测批大小")
    p_pred.add_argument("--nhits-fresh", action="store_true", help="强制重训 N-HiTS 并覆盖保存")
    p_pred.add_argument("--nhits-no-resume", action="store_true", help="不加载已保存 N-HiTS 模型")
    p_pred.add_argument("--bayes-nhits", action="store_true", help="启用 N-HiTS 贝叶斯调参")
    p_pred.add_argument("--bayes-nhits-calls", type=int, default=4, help="N-HiTS 贝叶斯搜索迭代次数")
    p_pred.add_argument("--bayes-nhits-recent", type=int, default=300, help="N-HiTS 贝叶斯调参使用的最近样本数")
    p_pred.add_argument("--nhits-backtest", action="store_true", help="对 N-HiTS 进行流式回测")
    p_pred.add_argument("--nhits-backtest-samples", type=int, default=300, help="N-HiTS 回测滑动样本数上限")
    p_pred.add_argument("--timesnet-fresh", action="store_true", help="强制重训 TimesNet 并覆盖保存")
    p_pred.add_argument("--timesnet-no-resume", action="store_true", help="不加载已保存 TimesNet 模型")
    p_pred.add_argument("--bayes-timesnet", action="store_true", help="启用 TimesNet 贝叶斯调参")
    p_pred.add_argument("--bayes-timesnet-calls", type=int, default=4, help="TimesNet 贝叶斯搜索迭代次数")
    p_pred.add_argument("--bayes-timesnet-recent", type=int, default=400, help="TimesNet 贝叶斯调参使用的最近样本数")
    p_pred.add_argument("--timesnet-rand", action="store_true", help="启用 TimesNet 随机调参兜底")
    p_pred.add_argument("--timesnet-rand-samples", type=int, default=80, help="TimesNet 随机搜索采样数")
    p_pred.add_argument("--timesnet-backtest", action="store_true", help="对 TimesNet 进行流式回测")
    p_pred.add_argument("--timesnet-backtest-samples", type=int, default=300, help="TimesNet 回测滑动样本数上限")
    p_pred.add_argument("--stack-bayes", action="store_true", help="对 stacking 元模型启用贝叶斯调参")
    p_pred.add_argument("--stack-bayes-calls", type=int, default=6, help="stacking 贝叶斯搜索迭代次数")
    p_pred.add_argument("--prophet-fresh", action="store_true", help="强制重训 Prophet 并覆盖保存")
    p_pred.add_argument("--prophet-no-resume", action="store_true", help="不加载已保存 Prophet 模型")
    p_pred.add_argument("--bayes-prophet", action="store_true", help="启用 Prophet 贝叶斯调参")
    p_pred.add_argument("--bayes-prophet-calls", type=int, default=4, help="Prophet 贝叶斯搜索迭代次数")
    p_pred.add_argument("--bayes-prophet-recent", type=int, default=200, help="Prophet 贝叶斯调参使用的最近样本数")
    p_pred.set_defaults(func=cmd_predict)

    p_pred_seq = sub.add_parser("predict-seq", help="使用 Transformer 序列模型进行下一期预测")
    p_pred_seq.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_pred_seq.add_argument("--recent", type=int, default=600, help="训练使用的最近 N 期样本（默认600）")
    p_pred_seq.add_argument("--window", type=int, default=20, help="滑窗长度")
    p_pred_seq.add_argument("--epochs", type=int, default=30, help="训练轮数")
    p_pred_seq.add_argument("--batch", type=int, default=64, help="batch size")
    p_pred_seq.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p_pred_seq.add_argument("--d-model", dest="d_model", type=int, default=96, help="模型隐层维度")
    p_pred_seq.add_argument("--nhead", type=int, default=4, help="多头注意力头数")
    p_pred_seq.add_argument("--layers", type=int, default=3, help="Transformer 层数")
    p_pred_seq.add_argument("--ff", type=int, default=192, help="前馈层维度")
    p_pred_seq.add_argument("--dropout", type=float, default=0.1, help="dropout")
    p_pred_seq.add_argument("--topk", type=int, default=3, help="输出前 topk 个号码及概率")
    p_pred_seq.add_argument("--seq-fresh", action="store_true", help="强制重训并覆盖保存")
    p_pred_seq.add_argument("--seq-no-resume", action="store_true", help="不加载已保存模型")
    p_pred_seq.set_defaults(func=cmd_predict_seq)

    p_pred_tft = sub.add_parser("predict-tft", help="使用 TFT 风格序列模型进行下一期预测")
    p_pred_tft.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_pred_tft.add_argument("--recent", type=int, default=800, help="训练使用的最近 N 期样本（默认800）")
    p_pred_tft.add_argument("--window", type=int, default=20, help="滑窗长度")
    p_pred_tft.add_argument("--epochs", type=int, default=30, help="训练轮数")
    p_pred_tft.add_argument("--batch", type=int, default=64, help="batch size")
    p_pred_tft.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p_pred_tft.add_argument("--d-model", dest="d_model", type=int, default=128, help="模型维度")
    p_pred_tft.add_argument("--nhead", type=int, default=4, help="多头注意力头数")
    p_pred_tft.add_argument("--layers", type=int, default=3, help="Transformer 层数")
    p_pred_tft.add_argument("--ff", type=int, default=256, help="前馈层维度")
    p_pred_tft.add_argument("--dropout", type=float, default=0.1, help="dropout")
    p_pred_tft.add_argument("--topk", type=int, default=3, help="输出前 topk 个号码及概率")
    p_pred_tft.add_argument("--tft-fresh", action="store_true", help="强制重训并覆盖保存")
    p_pred_tft.add_argument("--tft-no-resume", action="store_true", help="不加载已保存模型")
    p_pred_tft.set_defaults(func=cmd_predict_tft)

    p_tune_cat = sub.add_parser("tune-cat", help="贝叶斯优化 CatBoost 蓝球模型")
    p_tune_cat.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_tune_cat.add_argument("--recent", type=int, default=400, help="使用最近 N 期样本")
    p_tune_cat.add_argument("--window", type=int, default=10, help="滑窗长度")
    p_tune_cat.add_argument("--n-iter", dest="n_iter", type=int, default=15, help="贝叶斯搜索迭代次数")
    p_tune_cat.add_argument("--cv", type=int, default=3, help="交叉验证折数")
    p_tune_cat.add_argument("--topk", type=int, default=3, help="输出前 topk 个蓝球")
    p_tune_cat.set_defaults(func=cmd_tune_cat)

    p_cv_cat = sub.add_parser("cv-cat", help="CatBoost 位置模型滚动验证")
    p_cv_cat.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_cv_cat.add_argument("--recent", type=int, default=800, help="使用最近 N 期")
    p_cv_cat.add_argument("--window", type=int, default=10, help="CatBoost 滑窗")
    p_cv_cat.add_argument("--iter", type=int, default=200, help="CatBoost 迭代轮数")
    p_cv_cat.add_argument("--depth", type=int, default=6, help="CatBoost 深度")
    p_cv_cat.add_argument("--lr", type=float, default=0.1, help="CatBoost 学习率")
    p_cv_cat.add_argument("--train", type=int, default=300, help="每折训练集大小")
    p_cv_cat.add_argument("--test", type=int, default=20, help="每折测试集大小")
    p_cv_cat.add_argument("--step", type=int, default=20, help="窗口滑动步长")
    p_cv_cat.set_defaults(func=cmd_cv_cat)

    p_cv_tft = sub.add_parser("cv-tft", help="TFT 序列模型滚动验证")
    p_cv_tft.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_cv_tft.add_argument("--recent", type=int, default=800, help="使用最近 N 期")
    p_cv_tft.add_argument("--window", type=int, default=20, help="滑窗")
    p_cv_tft.add_argument("--epochs", type=int, default=30, help="训练轮数")
    p_cv_tft.add_argument("--batch", type=int, default=64, help="batch size")
    p_cv_tft.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p_cv_tft.add_argument("--d-model", dest="d_model", type=int, default=128, help="模型维度")
    p_cv_tft.add_argument("--nhead", type=int, default=4, help="多头注意力头数")
    p_cv_tft.add_argument("--layers", type=int, default=3, help="Transformer 层数")
    p_cv_tft.add_argument("--ff", type=int, default=256, help="前馈层维度")
    p_cv_tft.add_argument("--dropout", type=float, default=0.1, help="dropout")
    p_cv_tft.add_argument("--freq-window", dest="freq_window", type=int, default=50, help="频率特征滑窗")
    p_cv_tft.add_argument("--entropy-window", dest="entropy_window", type=int, default=50, help="熵特征滑窗")
    p_cv_tft.add_argument("--train", type=int, default=400, help="每折训练集大小")
    p_cv_tft.add_argument("--test", type=int, default=40, help="每折测试集大小")
    p_cv_tft.add_argument("--step", type=int, default=40, help="窗口滑动步长")
    p_cv_tft.set_defaults(func=cmd_cv_tft)

    p_train_all = sub.add_parser("train-all", help="一条命令依次训练 CatBoost / Transformer / TFT")
    p_train_all.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_train_all.add_argument("--sync", action="store_true", help="训练前先同步数据")
    p_train_all.add_argument("--fresh", action="store_true", help="强制重训所有模型并覆盖保存")
    p_train_all.add_argument("--recent", type=int, default=800, help="使用最近 N 期样本")
    # CatBoost
    p_train_all.add_argument("--no-cat", action="store_true", help="跳过 CatBoost 训练")
    p_train_all.add_argument("--cat-window", type=int, default=10, help="CatBoost 滑窗长度")
    p_train_all.add_argument("--cat-iter", type=int, default=300, help="CatBoost 迭代轮数")
    p_train_all.add_argument("--cat-depth", type=int, default=6, help="CatBoost 树深度")
    p_train_all.add_argument("--cat-lr", type=float, default=0.1, help="CatBoost 学习率")
    p_train_all.add_argument("--cat-fresh", action="store_true", help="强制重训 CatBoost 并覆盖保存")
    p_train_all.add_argument("--cat-no-resume", action="store_true", help="不加载已保存 CatBoost 模型")
    p_train_all.add_argument("--bayes-cat", action="store_true", help="开启 CatBoost 贝叶斯调参")
    p_train_all.add_argument("--bayes-cat-calls", type=int, default=8, help="CatBoost 贝叶斯搜索迭代次数")
    p_train_all.add_argument("--bayes-cat-cv", type=int, default=3, help="CatBoost 贝叶斯调参交叉验证折数")
    p_train_all.add_argument("--pbt-cat", action="store_true", help="使用 PBT 演化训练 CatBoost")
    # Transformer
    p_train_all.add_argument("--no-seq", action="store_true", help="跳过 Transformer 训练")
    p_train_all.add_argument("--seq-window", type=int, default=20, help="Transformer 滑窗")
    p_train_all.add_argument("--seq-epochs", type=int, default=20, help="Transformer 训练轮数")
    p_train_all.add_argument("--seq-d-model", dest="seq_d_model", type=int, default=96)
    p_train_all.add_argument("--seq-nhead", type=int, default=4)
    p_train_all.add_argument("--seq-layers", type=int, default=3)
    p_train_all.add_argument("--seq-ff", type=int, default=192)
    p_train_all.add_argument("--seq-dropout", type=float, default=0.1)
    p_train_all.add_argument("--seq-lr", type=float, default=1e-3)
    p_train_all.add_argument("--seq-fresh", action="store_true", help="强制重训 Transformer 并覆盖保存")
    p_train_all.add_argument("--seq-no-resume", action="store_true", help="不加载已保存 Transformer 模型")
    p_train_all.add_argument("--bayes-seq", action="store_true", help="开启 Transformer 贝叶斯调参")
    p_train_all.add_argument("--bayes-seq-calls", type=int, default=6, help="Transformer 贝叶斯搜索迭代次数")
    p_train_all.add_argument("--bayes-seq-recent", type=int, default=400, help="Transformer 贝叶斯调参使用的最近样本数")
    p_train_all.add_argument("--pbt-seq", action="store_true", help="使用 PBT 演化训练 Transformer")
    # TFT
    p_train_all.add_argument("--run-tft", action="store_true", help="开启 TFT 训练（耗时更长）")
    p_train_all.add_argument("--tft-window", type=int, default=20)
    p_train_all.add_argument("--tft-epochs", type=int, default=20)
    p_train_all.add_argument("--tft-batch", type=int, default=64)
    p_train_all.add_argument("--tft-lr", type=float, default=1e-3)
    p_train_all.add_argument("--tft-d-model", dest="tft_d_model", type=int, default=128)
    p_train_all.add_argument("--tft-nhead", type=int, default=4)
    p_train_all.add_argument("--tft-layers", type=int, default=3)
    p_train_all.add_argument("--tft-ff", type=int, default=256)
    p_train_all.add_argument("--tft-dropout", type=float, default=0.1)
    p_train_all.add_argument("--tft-freq-window", dest="tft_freq_window", type=int, default=50)
    p_train_all.add_argument("--tft-entropy-window", dest="tft_entropy_window", type=int, default=50)
    p_train_all.add_argument("--tft-fresh", action="store_true", help="强制重训 TFT 并覆盖保存")
    p_train_all.add_argument("--tft-no-resume", action="store_true", help="不加载已保存 TFT 模型")
    p_train_all.add_argument("--bayes-tft", action="store_true", help="开启 TFT 贝叶斯调参")
    p_train_all.add_argument("--bayes-tft-calls", type=int, default=4, help="TFT 贝叶斯搜索迭代次数")
    p_train_all.add_argument("--bayes-tft-recent", type=int, default=400, help="TFT 贝叶斯调参使用的最近样本数")
    p_train_all.add_argument("--tft-rand", action="store_true", help="启用 TFT 随机调参兜底")
    p_train_all.add_argument("--pbt-tft", action="store_true", help="使用 PBT 演化训练 TFT")
    # N-HiTS
    p_train_all.add_argument("--run-nhits", action="store_true", help="开启 N-HiTS 训练（和值+蓝球单变量）")
    p_train_all.add_argument("--nhits-input", type=int, default=60, help="N-HiTS 输入窗口")
    p_train_all.add_argument("--nhits-layers", type=int, default=2, help="N-HiTS 层数")
    p_train_all.add_argument("--nhits-blocks", type=int, default=1, help="N-HiTS block 数")
    p_train_all.add_argument("--nhits-steps", type=int, default=200, help="训练步数")
    p_train_all.add_argument("--nhits-lr", type=float, default=1e-3, help="学习率")
    p_train_all.add_argument("--nhits-fresh", action="store_true", help="强制重训 N-HiTS 并覆盖保存")
    p_train_all.add_argument("--nhits-no-resume", action="store_true", help="不加载已保存 N-HiTS 模型")
    p_train_all.add_argument("--bayes-nhits", action="store_true", help="开启 N-HiTS 贝叶斯调参")
    p_train_all.add_argument("--bayes-nhits-calls", type=int, default=4, help="N-HiTS 贝叶斯搜索迭代次数")
    p_train_all.add_argument("--bayes-nhits-recent", type=int, default=300, help="N-HiTS 贝叶斯调参使用的最近样本数")
    p_train_all.add_argument("--pbt-nhits", action="store_true", help="使用 PBT 演化训练 N-HiTS")
    # Prophet
    p_train_all.add_argument("--run-prophet", action="store_true", help="开启 Prophet 训练（和值/蓝球单变量）")
    p_train_all.add_argument("--prophet-fresh", action="store_true", help="强制重训 Prophet 并覆盖保存")
    p_train_all.add_argument("--prophet-no-resume", action="store_true", help="不加载已保存 Prophet 模型")
    p_train_all.add_argument("--bayes-prophet", action="store_true", help="开启 Prophet 贝叶斯调参")
    p_train_all.add_argument("--bayes-prophet-calls", type=int, default=4, help="Prophet 贝叶斯搜索迭代次数")
    p_train_all.add_argument("--bayes-prophet-recent", type=int, default=200, help="Prophet 贝叶斯调参使用的最近样本数")
    p_train_all.add_argument("--pbt-prophet", action="store_true", help="使用 PBT 演化训练 Prophet")
    # Blender
    p_train_all.add_argument("--run-blend", action="store_true", help="开启融合（蓝球/和值/红球位置动态加权）")
    p_train_all.add_argument("--blend-train", type=int, default=300, help="融合模型训练窗口大小")
    p_train_all.add_argument("--blend-test", type=int, default=30, help="融合模型验证测试集大小")
    p_train_all.add_argument("--blend-step", type=int, default=30, help="融合模型滚动步长")
    # TimesNet
    p_train_all.add_argument("--run-timesnet", action="store_true", help="开启 TimesNet 训练（和值+蓝球单变量）")
    p_train_all.add_argument("--timesnet-input", type=int, default=120, help="TimesNet 输入窗口")
    p_train_all.add_argument("--timesnet-hidden", type=int, default=64, help="TimesNet 隐层维度")
    p_train_all.add_argument("--timesnet-topk", type=int, default=5, help="TimesNet top_k")
    p_train_all.add_argument("--timesnet-steps", type=int, default=300, help="TimesNet 训练步数")
    p_train_all.add_argument("--timesnet-lr", type=float, default=1e-3, help="TimesNet 学习率")
    p_train_all.add_argument("--timesnet-dropout", type=float, default=0.1, help="TimesNet dropout")
    p_train_all.add_argument("--timesnet-fresh", action="store_true", help="强制重训 TimesNet 并覆盖保存")
    p_train_all.add_argument("--timesnet-no-resume", action="store_true", help="不加载已保存 TimesNet 模型")
    p_train_all.add_argument("--bayes-timesnet", action="store_true", help="开启 TimesNet 贝叶斯调参")
    p_train_all.add_argument("--bayes-timesnet-calls", type=int, default=4, help="TimesNet 贝叶斯搜索迭代次数")
    p_train_all.add_argument("--bayes-timesnet-recent", type=int, default=400, help="TimesNet 贝叶斯调参使用的最近样本数")
    p_train_all.add_argument("--timesnet-rand", action="store_true", help="启用 TimesNet 随机调参兜底")
    p_train_all.add_argument("--pbt-timesnet", action="store_true", help="使用 PBT 演化训练 TimesNet")
    p_train_all.add_argument("--pbt-generations", type=int, default=5, help="PBT 演化代数")
    p_train_all.add_argument("--pbt-steps", type=int, default=50, help="PBT 每代训练步数 (Cat/NHits/TimesNet)")
    p_train_all.add_argument("--pbt-epochs", type=int, default=1, help="PBT 每代训练轮数 (Seq/TFT/Prophet)")
    p_train_all.set_defaults(func=cmd_train_all)

    p_pbt = sub.add_parser("train-pbt", help="使用 PBT (Population Based Training) 演化训练模型")
    p_pbt.add_argument("--db", default="data/ssq.db", help="SQLite 数据库路径")
    p_pbt.add_argument("--recent", type=int, default=800, help="使用最近 N 期样本")
    p_pbt.add_argument("--model", type=str, required=True, choices=["seq", "cat", "tft", "nhits", "timesnet", "prophet"], help="目标模型")
    p_pbt.add_argument("--pop-size", type=int, default=4, help="种群大小")
    p_pbt.add_argument("--generations", type=int, default=10, help="演化代数")
    p_pbt.add_argument("--steps-per-gen", type=int, default=1, help="每代训练步数 (epochs)")
    p_pbt.add_argument("--n-jobs", type=int, default=1, help="并行线程数")
    p_pbt.add_argument("--window", type=int, default=20, help="窗口大小")
    p_pbt.set_defaults(func=cmd_train_pbt)

    return parser

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

