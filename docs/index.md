# Lottery AI Prediction System

Advanced lottery prediction system using Machine Learning and Deep Learning models.

## Features

- **Multi-Model Ensemble**: CatBoost, Transformer, TFT, N-HiTS, TimesNet, Prophet, GNN, RL
- **Population Based Training (PBT)**: Hyperparameter evolution
- **Chaos Theory Analysis**: Lyapunov exponent, correlation dimension
- **Real-time Dashboard**: Streamlit-based visualization

## Quick Start

```bash
# Sync data
python cli.py sync

# Train all models
python cli.py train-all

# Run predictions
python cli.py predict

# Launch dashboard
python cli.py dashboard
```

## Project Structure

```
lottery/
├── analyzer.py      # Data analysis & chaos theory
├── blender.py       # Model blending & ensemble
├── config.py        # Configuration management
├── features.py      # Feature engineering
├── ml_model.py      # CatBoost model
├── seq_model.py     # Transformer model
├── gnn_model.py     # Graph Neural Network
├── rl_model.py      # Reinforcement Learning
├── explainability.py # SHAP explanations
├── monte_carlo.py   # Monte Carlo simulation
└── diversity.py     # Ensemble diversity metrics
```

## API Reference

See the [API Reference](api/config.md) for detailed documentation.
