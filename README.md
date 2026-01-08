# 双色球爬取与混沌概率分析

一个简单的 Python 工具，用于：
- 爬取并结构化存储全部历史双色球开奖数据到 SQLite 数据库。
- 定期比对最新数据（基于系统时间与数据库已有期号），发现新增即抓取入库。
- 基于现代概率与混沌指标对数据做规律挖掘，输出频率、冷热、近似最大李雅普诺夫指数等指标。

## 快速开始
1. 安装依赖（建议使用虚拟环境）：
   ```bash
   pip install -r requirements.txt
   ```
2. 初始化并同步全部历史数据：
   ```bash
   python main.py sync --db data/ssq.db
   ```
   首次会抓取全量历史数据，后续重复执行会自动增量更新。
3. 运行分析：
   ```bash
   python main.py analyze --db data/ssq.db --recent 200
   ```
   `--recent` 可指定只分析最近 N 期。
4. 运行可视化 UI（Streamlit）：
   ```bash
   streamlit run app.py
   ```
   页面提供一键同步、频率/显著热冷号、遗漏、自相关、统计检验与推荐可视化；分析期数输入 0 表示使用全量数据。
5. 机器学习预测（CatBoost，位置多分类）：
   ```bash
   python main.py predict --db data/ssq.db --recent 400 --window 10 --topk 3
   ```
   可调参数：`--iter`(默认300)、`--depth`(6)、`--lr`(0.1)，输出红球各位置与蓝球的 Top-k 概率。
6. 序列模型预测（Transformer）：
   ```bash
   python main.py predict-seq --db data/ssq.db --recent 600 --window 20 --epochs 30 --topk 3
   ```
   可调参数：`--d-model`(96)、`--nhead`(4)、`--layers`(3)、`--ff`(192)、`--dropout`(0.1)、`--lr`(1e-3)。
7. TFT 风格序列模型预测：
   ```bash
   python main.py predict-tft --db data/ssq.db --recent 800 --window 20 --epochs 30 --topk 3
   ```
   可调参数：`--d-model`(128)、`--nhead`(4)、`--layers`(3)、`--ff`(256)、`--dropout`(0.1)、`--lr`(1e-3)。
8. CatBoost 位置模型滚动验证：
   ```bash
   python main.py cv-cat --db data/ssq.db --recent 800 --train 300 --test 20 --step 20
   ```
   输出各折红球位置 Top1 均值与蓝球 Top1 均值，便于评估泛化表现。
9. TFT 滚动验证：
   ```bash
   python main.py cv-tft --db data/ssq.db --recent 800 --train 400 --test 40 --step 40
   ```
   支持可调频率/熵滑窗与模型超参，用于评估时间分段下的稳定性。
   已加入多任务辅助头（和值回归、奇偶计数、跨度回归），帮助主任务收敛更稳。
10. 一条命令训练多模型（调度 CatBoost / Transformer / TFT）：
    ```bash
    # 默认跑 CatBoost 与 Transformer，TFT 需加 --run-tft，必要时加 --sync
    python main.py train-all --db data/ssq.db --recent 800 --sync --run-tft --run-nhits --run-prophet --run-timesnet
    ```
    可选参数：
    - CatBoost：`--cat-window 10 --cat-iter 300 --cat-depth 6 --cat-lr 0.1`
    - Transformer：`--seq-window 20 --seq-epochs 20 --seq-d-model 96 --seq-nhead 4 --seq-layers 3 --seq-ff 192 --seq-dropout 0.1 --seq-lr 1e-3`
    - TFT：`--run-tft --tft-window 20 --tft-epochs 20 --tft-d-model 128 --tft-nhead 4 --tft-layers 3 --tft-ff 256 --tft-dropout 0.1 --tft-freq-window 50 --tft-entropy-window 50`
    - N-HiTS（和值+蓝球单变量）：`--run-nhits --nhits-input 60 --nhits-layers 2 --nhits-blocks 1 --nhits-steps 200 --nhits-lr 1e-3`
    - Prophet（和值+蓝球单变量）：`--run-prophet`
    - TimesNet（和值+蓝球单变量）：`--run-timesnet --timesnet-input 120 --timesnet-hidden 64 --timesnet-topk 5 --timesnet-steps 300 --timesnet-lr 1e-3 --timesnet-dropout 0.1`
    - 融合（蓝球/和值/红球位置动态加权）：`--run-blend`（需至少两个基础模型开启，将自动做滚动融合与最新融合预测）

## 主要文件
- `main.py`：命令行入口，包含 `sync` 与 `analyze` 子命令。
- `lottery/scraper.py`：爬虫逻辑，默认从 500.com 历史页面解析全部开奖。
- `lottery/database.py`：SQLite schema 与增量写入。
- `lottery/analyzer.py`：概率统计与混沌指标计算。

## 数据源说明
- 主数据源：福彩官网 JSON 接口 `https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice`，一次拉取近 2000 期（约 2013 年至今）。
- 辅数据源：500.com 历史页 `https://datachart.500.com/ssq/history/inc/history.php?start=0001&end=9999`（2003-2009）以及分年段 `start=10001&end=10400` 等方式补齐 2010-2012。
- 解析包含：期号、开奖日期、6 红 1 蓝、销量/奖池/一等奖注数与奖金（字段可能因数据源有所缺失）。
- 若数据源结构变更，可在 `lottery/scraper.py` 中调整解析器。

## 混沌与概率分析概览
- 频率与 Dirichlet 平滑：估计红球 1-33、蓝球 1-16 的平滑出现概率。
- 冷热分析：输出近期窗口内的高频/低频号码。
- 近似最大李雅普诺夫指数：基于开奖和数序列的相空间重构（Rosenstein 思路）估计混沌程度。
- 熵与滑动窗口：评估短期随机性波动。
- 基础统计与假设检验：均值/标准差、红蓝球均匀性卡方统计、和值序列游程检验（独立同分布检验的近似）。
- 显著热/冷号标记：基于标准化残差（z≥2 / z≤-2）标出频率显著偏高或偏低的号码。
- 自相关分析：输出和值序列的低阶自相关系数（lag 1-5），用于识别序列相关性。
- 遗漏值分析：输出红/蓝球当前遗漏与历史最大遗漏，显示遗漏 Top3。
- 遗漏周期性提示：统计号码出现的间隔序列，若间隔波动极小（低 CV 且极差不大）则提示可能存在“准时出现”模式。
- 间隔统计输出：展示红/蓝球出现间隔 CV 最小的号码 Top5（含均值/标准差/最小/最大间隔），便于人工复核是否存在规律。
- 相空间重构：基于和值序列，支持 2-6 维、τ 可调的相空间散点可视化（可用于观察混沌吸引子形态）。
- 更严格的混沌检验：相关维数估计（简化 GP）、假最近邻比例 FNN、复现率/确定性(RR/DET) 指标。
- Apriori 关联规则：基于红+蓝的事务集（蓝球以前缀区分），输出支持度/置信度/提升度最高的规则（默认 sup>=0.01, conf>=0.2）。

## 注意
- 程序默认使用 SQLite 单机文件，方便无服务部署。
- 若需自动定时，可配合系统计划任务（Windows 任务计划、Linux cron）。
- 请遵守目标站点的 robots 与抓取频率要求，避免高频请求。

