# 双色球爬取、分析与多模型融合预测（含混沌指标）

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
5. 预测（多模型+融合+stacking+复式/杀号）：
   ```bash
   python main.py predict --db data/ssq.db --recent 400 --window 10 --topk 3 \
     --bayes-cat --bayes-seq --bayes-tft --bayes-nhits --bayes-timesnet --bayes-prophet \
     --stack-bayes
   ```
   - 可选流式回测：`--seq-backtest/--tft-backtest/--nhits-backtest/--timesnet-backtest`
   - 输出：基础模型、融合、XGBoost stacking、奇偶预测、和值标准差、约束复式、杀号。
6. 单模型序列预测：
   ```bash
   python main.py predict-seq --db data/ssq.db --recent 600 --window 20 --epochs 30 --topk 3
   python main.py predict-tft --db data/ssq.db --recent 800 --window 20 --epochs 30 --topk 3
   ```
7. 滚动验证：
   ```bash
   python main.py cv-cat --db data/ssq.db --recent 800 --train 300 --test 20 --step 20
   python main.py cv-tft --db data/ssq.db --recent 800 --train 400 --test 40 --step 40
   ```
8. 一键训练/调度：
   ```bash
   python main.py train-all --db data/ssq.db --sync --recent 800 \
     --run-tft --run-nhits --run-prophet --run-timesnet --run-blend
   ```
9. UI 快捷：
   ```bash
   streamlit run app.py
   ```

## 功能概览
- 数据与校验：全量/增量抓取；入库前校验红 6 不重复且 1-33，蓝 1 且 1-16。
- 分析：频率/热冷、基础统计、卡方/游程、自相关、遗漏/周期、相空间、相关维数、FNN、RR/DET、Apriori。
- 特征：AC 值、和值尾、跨度、奇偶/质合/大小比、遗漏、农历、周几等。
- 预测模型（GPU 优先，含 FocalLoss/组合哈希等增强）：
  - CatBoost 位置；Transformer；TFT（多任务：和值/奇偶/跨度）
  - N-HiTS / TimesNet / Prophet（和值+蓝球）
  - 奇偶模型（红球奇数个数）；和值标准差模型（用于约束）
- 融合/stacking：
  - 动态加权融合（蓝/红/和值）
  - XGBoost 元模型（概率向量特征，支持贝叶斯调参）
- 约束与推荐：
  - 基于高概率红/蓝 + 和值均值±预测标准差 + 奇偶预测生成复式
  - 杀号：概率 < 0.01% 的红/蓝
- 回测：
  - CatBoost/TFT 滚动验证
  - Transformer/TFT/N-HiTS/TimesNet 流式回测（IterableDataset）

## 数据源
- 主：福彩官网 JSON（约 2013~今） `https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice`
- 辅：500.com 历史页与分年段补齐 2003~2012

## 目录速览
- `main.py`：CLI、调度、融合/stacking、复式与杀号
- `lottery/scraper.py`：爬虫
- `lottery/database.py`：Schema 与入库校验
- `lottery/analyzer.py`：统计/混沌/Apriori
- `lottery/ml_model.py` 等：CatBoost/Transformer/TFT/N-HiTS/TimesNet/Prophet
- `lottery/odd_model.py`：奇偶数预测
- `lottery/sum_model.py`：和值标准差预测
- `lottery/blender.py`：融合、stacking、复式、杀号

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

