from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from lottery import analyzer, database, scraper


def sync_db(db_path: Path) -> str:
    database.init_db(db_path)
    all_draws = scraper.fetch_all_draws()
    with database.get_conn(db_path) as conn:
        existing = {row["issue"] for row in conn.execute("SELECT issue FROM draws")}
        to_insert = scraper.filter_new_draws(all_draws, existing)
        if not existing:
            to_insert = all_draws
        if not to_insert:
            return "数据库已是最新，无需更新。"
        affected = database.upsert_draws(conn, to_insert)
        latest_issue = max(d.issue for d in to_insert)
        return f"新增/更新 {affected} 期，最新期号 {latest_issue}"


def load_df(db_path: Path, recent: int | None) -> pd.DataFrame:
    with database.get_conn(db_path) as conn:
        return analyzer.load_dataframe(conn, recent=recent)


def main() -> None:
    st.set_page_config(page_title="双色球分析", layout="wide")
    st.title("双色球爬取与分析")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        db_path_str = st.text_input("数据库路径", value="data/ssq.db")
        db_path = Path(db_path_str)
    with col2:
        recent_input = st.number_input("分析最近 N 期（0=全量）", min_value=0, step=50, value=200, help="填写 0 表示分析全量历史数据")
        recent = int(recent_input) if recent_input > 0 else None
    with col3:
        if st.button("同步最新数据", type="primary"):
            with st.spinner("同步中..."):
                msg = sync_db(db_path)
            st.success(msg)
            st.session_state["last_sync_msg"] = msg

    if not db_path.exists():
        last_msg = st.session_state.get("last_sync_msg")
        if last_msg:
            st.info(f"最近同步结果：{last_msg}")
        st.info("数据库不存在，请先点击“同步最新数据”。")
        st.stop()

    df = load_df(db_path, recent=recent)
    if df.empty:
        st.warning("数据库为空，请先同步数据。")
        st.stop()

    report = analyzer.analyze(df, entropy_window=60)

    st.subheader("概览")
    cols = st.columns(4)
    cols[0].metric("样本期数", report["count"])
    cols[1].metric("最新期号", report["latest_issue"])
    cols[2].metric("近期熵", f"{report['entropy_recent']:.3f}" if report["entropy_recent"] else "N/A")
    cols[3].metric("最大李雅普诺夫", f"{report['lyapunov']:.3f}" if report["lyapunov"] else "N/A")

    # 频率分布
    st.subheader("频率与显著热/冷号")
    freq_tabs = st.tabs(["红球频率", "蓝球频率"])
    red_probs = pd.DataFrame({"num": list(report["probs_red"].keys()), "prob": list(report["probs_red"].values())})
    blue_probs = pd.DataFrame({"num": list(report["probs_blue"].keys()), "prob": list(report["probs_blue"].values())})

    sig = report.get("significant", {})
    red_sig_hot = set(sig.get("red", {}).get("hot", []))
    red_sig_cold = set(sig.get("red", {}).get("cold", []))
    blue_sig_hot = set(sig.get("blue", {}).get("hot", []))
    blue_sig_cold = set(sig.get("blue", {}).get("cold", []))

    with freq_tabs[0]:
        red_probs["tag"] = red_probs["num"].apply(
            lambda x: "显著热" if x in red_sig_hot else ("显著冷" if x in red_sig_cold else "常规")
        )
        fig = px.bar(red_probs, x="num", y="prob", color="tag", color_discrete_map={"显著热": "red", "显著冷": "blue", "常规": "gray"})
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"显著热红号: {sorted(red_sig_hot)} | 显著冷红号: {sorted(red_sig_cold)}")

    with freq_tabs[1]:
        blue_probs["tag"] = blue_probs["num"].apply(
            lambda x: "显著热" if x in blue_sig_hot else ("显著冷" if x in blue_sig_cold else "常规")
        )
        fig = px.bar(blue_probs, x="num", y="prob", color="tag", color_discrete_map={"显著热": "red", "显著冷": "blue", "常规": "gray"})
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"显著热蓝号: {sorted(blue_sig_hot)} | 显著冷蓝号: {sorted(blue_sig_cold)}")

    st.subheader("遗漏分析")
    omission = report.get("omission", {})
    red_om = omission.get("red", {})
    blue_om = omission.get("blue", {})
    red_df = pd.DataFrame(
        [{"num": n, "current": v["current"], "max": v["max"]} for n, v in red_om.items()]
    ).sort_values("current", ascending=False)
    blue_df = pd.DataFrame(
        [{"num": n, "current": v["current"], "max": v["max"]} for n, v in blue_om.items()]
    ).sort_values("current", ascending=False)
    col_r, col_b = st.columns(2)
    with col_r:
        st.markdown("**红球当前遗漏 Top10**")
        st.dataframe(red_df.head(10), hide_index=True)
    with col_b:
        st.markdown("**蓝球当前遗漏 Top10**")
        st.dataframe(blue_df.head(10), hide_index=True)

    st.subheader("和值序列相关性与检验")
    ac = report.get("autocorr", {})
    if ac:
        ac_df = pd.DataFrame({"lag": list(ac.keys()), "corr": list(ac.values())})
        fig_ac = px.bar(ac_df, x="lag", y="corr")
        st.plotly_chart(fig_ac, use_container_width=True)
    runs = report.get("runs_test_sum")
    if runs:
        st.write(f"游程检验: runs={runs['runs']}, z={runs['z']:.3f}, p≈{runs['p']:.3f}")

    st.subheader("混沌检验（和值序列）")
    chaos = report.get("chaos", {})
    col_c1, col_c2, col_c3 = st.columns(3)
    if chaos.get("corr_dim"):
        cd = chaos["corr_dim"]
        msg = "; ".join(f"m{m}:{cd[m]:.2f}" for m in sorted(cd.keys()))
        col_c1.markdown(f"**相关维数估计**: {msg}")
    else:
        col_c1.markdown("**相关维数估计**: N/A")
    if chaos.get("fnn"):
        fnn = chaos["fnn"]
        msg = "; ".join(f"m{m}->{m+1}:{fnn[m]:.1f}%" for m in sorted(fnn.keys()))
        col_c2.markdown(f"**假最近邻比例**: {msg}")
    else:
        col_c2.markdown("**假最近邻比例**: N/A")
    if chaos.get("recur"):
        r = chaos["recur"]
        col_c3.markdown(f"**复现率/DET**: RR={r.get('rr', float('nan')):.4f}, DET={r.get('det', float('nan')):.4f} (eps={r.get('eps', float('nan')):.4f})")
    else:
        col_c3.markdown("**复现率/DET**: N/A")

    st.subheader("相空间重构（和值序列）")
    col_dim, col_tau, col_max = st.columns(3)
    dim = col_dim.slider("维度", min_value=2, max_value=6, value=3, step=1)
    tau = col_tau.slider("延迟 τ", min_value=1, max_value=5, value=1, step=1)
    max_pts = col_max.slider("展示点数上限", min_value=200, max_value=3000, value=1000, step=100)
    sums_series = df[["red1", "red2", "red3", "red4", "red5", "red6", "blue"]].sum(axis=1).to_numpy()
    phase_df = analyzer.phase_space(sums_series, dim=dim, tau=tau, max_points=max_pts)
    if phase_df.empty:
        st.info("数据不足，无法重构相空间。")
    else:
        if dim >= 3:
            fig_ps = px.scatter_3d(phase_df, x="x1", y="x2", z="x3", opacity=0.7, size_max=4)
            st.plotly_chart(fig_ps, use_container_width=True)
        else:
            fig_ps = px.scatter(phase_df, x="x1", y="x2", opacity=0.7)
            st.plotly_chart(fig_ps, use_container_width=True)

    st.subheader("基础统计")
    stats = report.get("basic_stats", {})
    chi = report.get("chi_square", {})
    cols_stats = st.columns(3)
    cols_stats[0].markdown(f"- 红均值 {stats.get('red_mean', float('nan')):.2f} ± {stats.get('red_std', float('nan')):.2f}")
    cols_stats[1].markdown(f"- 蓝均值 {stats.get('blue_mean', float('nan')):.2f} ± {stats.get('blue_std', float('nan')):.2f}")
    cols_stats[2].markdown(f"- 和值均值 {stats.get('sum_mean', float('nan')):.2f} ± {stats.get('sum_std', float('nan')):.2f}")
    st.markdown(
        f"卡方检验：红 stat={chi.get('red', {}).get('stat', float('nan')):.2f} (dof={chi.get('red', {}).get('dof', 0)}), "
        f"蓝 stat={chi.get('blue', {}).get('stat', float('nan')):.2f} (dof={chi.get('blue', {}).get('dof', 0)})"
    )

    st.subheader("推荐组合")
    suggestion = report["suggestion"]
    st.success(f"红 {suggestion['reds']} + 蓝 {suggestion['blue']}")

    st.subheader("关联规则 (Apriori)")
    rules = report.get("rules", [])
    if rules:
        df_rules = pd.DataFrame(
            [
                {
                    "前件": ", ".join(r["antecedent"]),
                    "后件": ", ".join(r["consequent"]),
                    "支持度": round(r["support"], 3),
                    "置信度": round(r["confidence"], 3),
                    "提升度": round(r["lift"], 3),
                }
                for r in rules[:20]
            ]
        )
        st.dataframe(df_rules, hide_index=True)
    else:
        st.write("暂无满足阈值的规则（默认 sup>=0.01, conf>=0.2）。")


if __name__ == "__main__":
    main()

