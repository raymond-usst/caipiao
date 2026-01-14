# 🎱 Lottery AI - Advanced Prediction System / 双色球 AI 预测系统

> **State-of-the-art Deep Learning & Machine Learning for Lottery Prediction**
> **基于深度学习与机器学习的前沿彩票预测系统**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Introduction / 简介

This project is a comprehensive AI-powered system designed to analyze and predict "Dual Color Ball" (Union Lotto) lottery numbers. It leverages a diverse ensemble of **Deep Learning**, **Statistical**, and **Chaos Theory** models to capture both linear patterns and non-linear chaotic dynamics in lottery data.

本项目是一个综合性的 AI 智能系统，旨在分析和预测“双色球”彩票号码。它利用**深度学习**、**统计学**和**混沌理论**等多种模型集成，捕捉彩票数据中的线性规律与非线性混沌动态。

Unlike traditional statistical tools, this system aims for **principled uncertainty quantification** and **adaptive learning** using advanced techniques like Population Based Training (PBT), Bayesian Neural Networks, and Meta-Learning.

与传统统计工具不同，本系统追求**原理性的不确定性量化**和**自适应学习**，采用了基于群体的训练 (PBT)、贝叶斯神经网络和元学习等前沿技术。

---

## ✨ Key Features / 核心功能

### 🧠 Advanced Modeling / 高级建模
*   **Ensemble Learning (集成学习)**: Blends predictions from **10+ models** (CatBoost, Transformer, LSTM, DeepCross, TFT, Prophet, N-HiTS, TimesNet).
*   **Graph Neural Networks (图神经网络)**: **GAT (Graph Attention Network)** to model co-occurrence relationships between numbers.
*   **Reinforcement Learning (强化学习)**: Policy Gradient based agent optimizing directly for reward (hits).
*   **Chaos Theory (混沌理论)**: **Echo State Networks (ESN)** and chaotic features (Lyapunov Exponent, Correlation Dimension).
*   **Bayesian Inference (贝叶斯推断)**: **BNN (Bayesian Neural Network)** for epistemic uncertainty quantification.
*   **Meta-Learning (元学习)**: MAML-style fast adaptation with **Regime Detection** for shifting data distributions.

### ⚙️ Robust Engineering / 稳健工程
*   **Population Based Training (PBT)**: Auto-tuning hyperparameters and evolving model populations during training.
*   **Principled Uncertainty (不确定性)**: **Conformal Prediction** for guaranteed coverage intervals and **Calibration** (Temperature/Platt Scaling).
*   **High Performance (高性能)**: **Polars** based feature engineering and **Numba** JIT acceleration.
*   **Explainability (可解释性)**: **SHAP** values for feature importance analysis.

### 📊 Visualization & Ops / 可视化与运维
*   **Predictive Dashboard (预测仪表盘)**: Interactive **Streamlit** app for real-time analysis and training monitoring.
*   **Prediction Audit (预测审计)**: Full audit trail of all predictions with timestamping.
*   **Comprehensive Evaluation (全方位评估)**: Time-Series Cross-Validation and Proper Scoring Rules (Brier, CRPS).

---

## 🚀 Quick Start / 快速开始

### 1. Installation / 安装
```bash
# Clone the repository / 克隆仓库
git clone https://github.com/yourusername/lottery_ai.git
cd lottery_ai

# Install dependencies / 安装依赖
pip install -r requirements.txt
```

### 2. Data Sync / 数据同步
Fetch the latest lottery data from the internet.
从互联网获取最新的彩票数据。
```bash
python main.py sync
```

### 3. Training / 训练模型
Train all models in the ensemble (supports parallel execution).
训练集成中的所有模型（支持并行执行）。
```bash
python main.py train-all
```

### 4. Prediction / 预测
Generate predictions for the next draw.
生成下一期预测号码。
```bash
python main.py predict
```

### 5. Dashboard / 启动仪表盘
Launch the interactive visualization interface.
启动交互式可视化界面。
```bash
python dashboard.py
```

---

## 🛠️ CLI Arguments - `train-all` Parameters

Use `python main.py train-all [ARGS]` to customize training.
所有参数均为可选，支持详细配置。

### 1. Global & Data / 全局与数据
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--db` | `data/ssq.db` | SQLite 数据库路径 |
| `--sync` | False | 训练前自动同步最新数据 |
| `--fresh` | False | **强制重训所有模型**（忽略断点，清空旧模型） |
| `--recent` | 800 | 使用最近 N 期数据进行训练（建议 >= 800） |

### 2. CatBoost Models
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--no-cat` | False | 跳过 CatBoost 训练 |
| `--cat-window` | 10 | 特征滑窗长度 |
| `--cat-iter` | 300 | 迭代轮数 (Trees) |
| `--cat-depth` | 6 | 树深度 |
| `--cat-lr` | 0.1 | 学习率 |
| `--cat-fresh` | False | 强制重训 CatBoost |
| `--cat-no-resume`| False | 不加载已保存模型 |
| `--bayes-cat` | False | 开启贝叶斯超参数优化 |
| `--bayes-cat-calls`| 8 | 贝叶斯搜索次数 |
| `--pbt-cat` | False | 开启 PBT 演化训练 |

### 3. Transformer (Seq2Seq)
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--no-seq` | False | 跳过 Transformer 训练 |
| `--seq-window` | 20 | 序列滑窗长度 |
| `--seq-epochs` | 20 | 训练轮数 |
| `--seq-d-model` | 96 | 模型维度 |
| `--seq-lr` | 1e-3 | 学习率 |
| `--bayes-seq` | False | 开启 Transformer 贝叶斯调参 |
| `--pbt-seq` | False | 开启 PBT 演化训练 |

### 4. TFT (Temporal Fusion Transformer)
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--run-tft` | False | **开启 TFT 训练**（默认关闭） |
| `--tft-window` | 20 | 输入窗口长度 |
| `--tft-epochs` | 20 | 训练轮数 |
| `--tft-batch` | 64 | Batch Size |
| `--tft-lr` | 1e-3 | 学习率 |
| `--bayes-tft` | False | 开启 TFT 贝叶斯调参 |
| `--pbt-tft` | False | 开启 PBT 演化训练 |

### 5. Advanced Time Series (N-HiTS / TimesNet / Prophet)
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--run-nhits` | False | 开启 N-HiTS 训练 |
| `--nhits-steps` | 200 | 训练步数 |
| `--run-timesnet` | False | 开启 TimesNet 训练 |
| `--timesnet-steps`| 300 | 训练步数 |
| `--run-prophet` | False | 开启 Prophet 训练 |
| `--bayes-*` | False | 对应模型的贝叶斯调参 (e.g. `--bayes-nhits`) |
| `--pbt-*` | False | 对应模型的 PBT 演化训练 (e.g. `--pbt-timesnet`) |

### 6. PBT & Evolution Settings / PBT 演化配置
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--pbt-generations`| 5 | 演化代数（Generations） |
| `--pbt-steps` | 50 | 每代训练步数 (Cat/N-HiTS/TimesNet) |
| `--pbt-epochs` | 1 | 每代训练轮数 (Seq/TFT/Prophet) |

### 7. Blender (Ensemble) / 模型融合
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--run-blend` | False | **开启模型融合**（推荐开启） |
| `--blend-train` | 300 | 融合模型训练集窗口 |
| `--blend-test` | 30 | 融合模型验证测试集大小 |
| `--blend-step` | 30 | 滚动验证步长 |

---

## 🧠 Model Zoo / 模型库

| Model Type | Models Included | Description (CN) |
|:---|:---|:---|
| **Tree-Based** | `CatBoost` | Gradient boosting decision trees, handles categorical features well. / 梯度提升树，擅长处理分类特征。 |
| **Sequence** | `Transformer`, `LSTM` | Captures long-term temporal dependencies. / 捕捉长期时间依赖关系。 |
| **Time-Series** | `TFT`, `N-HiTS`, `TimesNet`, `Prophet` | Specialized operational research & forecasting models. / 专业的运筹学与预测模型。 |
| **Graph** | `LotteryGNN` (GAT) | Models number co-occurrences as a graph. / 将号码共现关系建模为图结构。 |
| **Chaos** | `ESN` (Echo State Network) | Reservoir computing for chaotic dynamics. / 用于混沌动态的储备池计算。 |
| **Probabilistic** | `LotteryBNN` | Bayesian weights for uncertainty estimation. / 用于不确定性估计的贝叶斯权重。 |
| **Adaptive** | `MetaLearning` | Detects regime shifts and fast-adapts. / 检测机制转换并快速适应。 |
| **RL** | `PolicyGradient` | Optimizes directly for lottery reward rules. / 直接针对彩票奖励规则进行优化。 |

---

## 📂 Project Structure / 项目结构

```
lottery/
├── cli.py               # Command Line Interface / 命令行入口
├── dashboard.py         # Streamlit Dashboard / 仪表盘
├── lottery/
│   ├── engine/          # Core Engines (Trainer, Predictor) / 核心引擎
│   ├── features.py      # Feature Engineering (Polars) / 特征工程
│   ├── blender.py       # Model Fusion / 模型融合
│   ├── *_model.py       # Individual Model Implementations / 各个模型实现
│   ├── pbt.py           # Population Based Training / 基于群体的训练
│   ├── conformal.py     # Conformal Prediction / 适形预测
│   └── calibration.py   # Probability Calibration / 概率校准
└── docs/                # Documentation / 文档
```

---

## ⚠️ Disclaimer / 免责声明

**English**:
This project is for **research and educational purposes only**. Lottery entails significant risk, and buying lottery tickets is a form of gambling. The predictions generated by this system are based on historical data and probabilistic models, which **cannot guarantee** winning results. The author assumes no responsibility for any financial losses incurred by using this software. Please play responsibly.

**中文**:
本项目仅供**研究和教育用途**。彩票具有巨大的风险，购买彩票属于博彩行为。本系统生成的预测基于历史数据和概率模型，**不能保证**中奖。对于使用本软件造成的任何经济损失，作者不承担任何责任。请理性购彩，量力而行。
