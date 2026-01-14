from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml
from lottery.utils.logger import logger

@dataclass
class DatabaseConfig:
    path: str = "data/ssq.db"

@dataclass
class CatBoostConfig:
    enabled: bool = True
    window: int = 10
    iterations: int = 300
    depth: int = 6
    learning_rate: float = 0.1
    resume: bool = True
    fresh: bool = False
    
@dataclass
class SeqConfig:
    enabled: bool = True
    window: int = 20
    epochs: int = 20
    batch_size: int = 64
    d_model: int = 96
    nhead: int = 4
    num_layers: int = 3
    ff: int = 192
    dropout: float = 0.1
    lr: float = 1e-3
    resume: bool = True
    fresh: bool = False

@dataclass
class TFTConfig:
    enabled: bool = False
    window: int = 20
    epochs: int = 20
    batch_size: int = 64
    d_model: int = 128
    nhead: int = 4
    layers: int = 3
    ff: int = 256
    dropout: float = 0.1
    lr: float = 1e-3
    freq_window: int = 50
    entropy_window: int = 50
    resume: bool = True
    fresh: bool = False

@dataclass
class NHitsConfig:
    enabled: bool = False
    input_size: int = 60
    n_layers: int = 2
    n_blocks: int = 1
    max_steps: int = 200
    learning_rate: float = 1e-3
    resume: bool = True
    fresh: bool = False
    backtest: bool = False
    backtest_samples: int = 300

@dataclass
class TimesNetConfig:
    enabled: bool = False
    input_size: int = 120
    hidden_size: int = 64
    top_k: int = 5
    max_steps: int = 300
    learning_rate: float = 1e-3
    dropout: float = 0.1
    resume: bool = True
    fresh: bool = False
    backtest: bool = False
    backtest_samples: int = 300

@dataclass
class ProphetConfig:
    enabled: bool = False
    resume: bool = True
    fresh: bool = False

@dataclass
class BlendConfig:
    enabled: bool = True
    train_size: int = 300
    test_size: int = 30
    step: int = 30
    alpha: float = 0.3
    l1_ratio: float = 0.1

@dataclass
class GNNConfig:
    enabled: bool = False
    hidden_channels: int = 64
    num_layers: int = 2
    dropout: float = 0.5
    lr: float = 0.01
    epochs: int = 200
    top_k_neighbors: int = 5
    resume: bool = True
    fresh: bool = False

@dataclass
class RLConfig:
    enabled: bool = False
    window: int = 10
    hidden_size: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    episodes: int = 1000
    batch_size: int = 32
    resume: bool = True
    fresh: bool = False

@dataclass
class ESNConfig:
    enabled: bool = False
    window: int = 20
    reservoir_size: int = 500
    spectral_radius: float = 0.95
    leak_rate: float = 0.3
    resume: bool = True
    fresh: bool = False

@dataclass
class BNNConfig:
    enabled: bool = False
    window: int = 10
    hidden_dim: int = 64
    epochs: int = 100
    resume: bool = True
    fresh: bool = False

@dataclass
class MetaConfig:
    enabled: bool = False
    hidden_dim: int = 32
    outer_lr: float = 0.001
    inner_lr: float = 0.01
    resume: bool = True
    fresh: bool = False

@dataclass
class TrainAllConfig:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    recent: int = 800
    sync: bool = False
    cat: CatBoostConfig = field(default_factory=CatBoostConfig)
    seq: SeqConfig = field(default_factory=SeqConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)
    nhits: NHitsConfig = field(default_factory=NHitsConfig)
    timesnet: TimesNetConfig = field(default_factory=TimesNetConfig)
    prophet: ProphetConfig = field(default_factory=ProphetConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    esn: ESNConfig = field(default_factory=ESNConfig)
    bnn: BNNConfig = field(default_factory=BNNConfig)
    meta: MetaConfig = field(default_factory=MetaConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)

def load_config(path: str) -> TrainAllConfig:
    """Load configuration from a YAML file."""
    p = Path(path)
    if not p.exists():
        logger.warning(f"Config file {path} not found. Using defaults.")
        return TrainAllConfig()
    
    with open(p, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Helper to load nested dataclasses
    # This is a simple implementation. For production, consider using dacite or omegaconf
    def from_dict(cls, data):
        if not data:
            return cls()
        valid_keys = cls.__annotations__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    cfg = TrainAllConfig(
        db=from_dict(DatabaseConfig, data.get('db')),
        recent=data.get('recent', 800),
        sync=data.get('sync', False),
        cat=from_dict(CatBoostConfig, data.get('cat')),
        seq=from_dict(SeqConfig, data.get('seq')),
        tft=from_dict(TFTConfig, data.get('tft')),
        nhits=from_dict(NHitsConfig, data.get('nhits')),
        timesnet=from_dict(TimesNetConfig, data.get('timesnet')),
        prophet=from_dict(ProphetConfig, data.get('prophet')),
        gnn=from_dict(GNNConfig, data.get('gnn')),
        rl=from_dict(RLConfig, data.get('rl')),
        esn=from_dict(ESNConfig, data.get('esn')),
        bnn=from_dict(BNNConfig, data.get('bnn')),
        meta=from_dict(MetaConfig, data.get('meta')),
        blend=from_dict(BlendConfig, data.get('blend')),
    )
    return cfg
