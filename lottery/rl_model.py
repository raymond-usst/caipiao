from __future__ import annotations
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from lottery.utils.logger import logger
from lottery.engine.predictor import BasePredictor
from lottery.config import RLConfig
from lottery.pbt import Member, ModelAdapter

# Policy Network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim=33):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Output probability for each number (1-33)
        return torch.sigmoid(self.fc2(x))

class LotteryEnv:
    def __init__(self, data: np.ndarray, window=10):
        self.data = data # [N, 7]
        self.window = window
        self.current_step = window
        
    def reset(self):
        self.current_step = self.window
        return self._get_state()
        
    def _get_state(self):
        # Flatten last 'window' draws as state
        # Only using red balls for simplicity? Or full?
        # Let's use Reds only for state size simplicity in this demo
        # data: [N, 6] (red only)
        # state: [window * 6]
        slice_data = self.data[self.current_step - self.window : self.current_step, :6]
        return slice_data.flatten()
        
    def step(self, action_probs):
        # Action: selection of 6 numbers.
        # But for RL training with Policy Gradient, we sample actions or use probs.
        # Reward: How many matches with next draw?
        
        if self.current_step >= len(self.data):
            return None, 0, True, {}
            
        target = self.data[self.current_step, :6] # True Red Balls
        
        # Simple reward: sum of probabilities assigned to the true numbers?
        # Or if we sample: count matches.
        # Let's use a dense reward for training stability: sum of log probs of true targets (Maximize Likelihood)
        # This basically degrades to Supervised Learning if we just maximize log_prob of target.
        # To make it "RL", let's sample 6 numbers and reward based on intersection size.
        
        # Sample 6 numbers without replacement based on probs
        # To make it differentiable? REINFORCE uses log_prob of sampled action * reward.
        
        # We will assume 'action_probs' is a tensor from policy.
        
        # Mock step just to advance
        self.current_step += 1
        done = self.current_step >= len(self.data)
        
        # Calculate matches (offline logic usually)
        return self._get_state(), target, done
    
class RLPredictor(BasePredictor):
    def __init__(self, config: RLConfig):
        super().__init__(config)
        self.cfg = config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = getattr(config, "window", 10)

    def train(self, df: pd.DataFrame) -> None:
        logger.info("Training RL Policy...")
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols].to_numpy(dtype=float)
        # Normalize
        data = data / 33.0 
        
        save_dir = "models/RL"
        loaded = False
        if self.cfg.resume and not self.cfg.fresh:
            if self.load(save_dir):
                logger.info(f"Resumed RL model from {save_dir}")
                loaded = True
        
        if not loaded:
            state_dim = self.window * 6
            self.model = PolicyNet(state_dim, self.cfg.hidden_size, action_dim=33).to(self.device)
            if len(data) <= self.window:
                logger.warning(f"Not enough data for RL training ({len(data)} <= {self.window}). Skipping training.")
                return

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        
        # REINFORCE-like training loop
        # For efficiency, we can batch process windows
        
        self.model.train()
        
        # Running baseline for variance reduction
        running_baseline = 0.0
        baseline_alpha = 0.1
        entropy_coef = 0.01  # Entropy regularization coefficient
        
        for ep in range(self.cfg.episodes):
            # Randomly sample batch
            indices = np.random.randint(self.window, len(data), size=self.cfg.batch_size)
            
            log_probs = []
            rewards = []
            entropies = []
            
            for idx in indices:
                state = data[idx - self.window : idx].flatten()
                target_raw = df[red_cols].iloc[idx].to_numpy(dtype=int)  # 1-33
                
                state_t = torch.FloatTensor(state).to(self.device)
                probs = self.model(state_t)  # [33]
                
                # Sample action using Bernoulli
                m = torch.distributions.Bernoulli(probs)
                action = m.sample()
                
                # Entropy for exploration
                entropy = m.entropy().sum()
                entropies.append(entropy)
                
                # Reward structure:
                # +10 for each match, +20 bonus for 3+ matches
                # -1 for each selected number (cost)
                selected_indices = torch.nonzero(action).squeeze(-1).cpu().numpy() + 1
                matches = len(set(selected_indices) & set(target_raw))
                cost = len(selected_indices)
                
                reward = matches * 10.0 - cost * 1.0
                if matches >= 3:
                    reward += 20.0  # Bonus for hitting 3+
                
                log_prob = m.log_prob(action).sum()
                log_probs.append(log_prob)
                rewards.append(reward)
            
            # Update running baseline
            avg_reward = np.mean(rewards)
            running_baseline = baseline_alpha * avg_reward + (1 - baseline_alpha) * running_baseline
            
            # Compute advantage (reward - baseline)
            advantages = [r - running_baseline for r in rewards]
            
            # Update policy
            optimizer.zero_grad()
            policy_loss = []
            for lp, adv in zip(log_probs, advantages):
                policy_loss.append(-lp * adv)
            
            # Add entropy bonus (maximize entropy = minimize negative entropy)
            entropy_bonus = -entropy_coef * torch.stack(entropies).mean()
            
            loss = torch.stack(policy_loss).mean() + entropy_bonus
            loss.backward()
            optimizer.step()
            
            if ep % 100 == 0:
                avg_reward = np.mean(rewards)
                logger.info(f"RL Episode {ep}: Avg Reward {avg_reward:.2f}")

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self.model is None:
             from lottery.utils.exceptions import ModelNotFittedError
             raise ModelNotFittedError("RL model not trained")
             
        self.model.eval()
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = df[red_cols].to_numpy(dtype=float) / 33.0
        
        if len(data) < self.window:
             # Padding or error
             # Silent padding to avoid log spam during backtesting/training
             data = np.pad(data, ((self.window - len(data), 0), (0,0)), 'constant')
             
        state = data[-self.window:].flatten()
        state_t = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            probs = self.model(state_t).cpu().numpy()
            
        # Top 6
        top_indices = np.argsort(probs)[::-1]
        top_reds = [(i + 1, float(probs[i])) for i in top_indices[:6]]
        
        # Dummy blue
        blue_counts = df["blue"].value_counts()
        best_blue = blue_counts.idxmax() if not blue_counts.empty else 1

        return {
            "flat_reds": top_reds,
            "blue": [(best_blue, 1.0)],
            "sum_pred": 100
        }

    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / "rl.pt"
        torch.save(self.model.state_dict(), file_path)

    def load(self, save_dir: str) -> bool:
        path = Path(save_dir) / "rl.pt"
        if not path.exists(): return False
        try:
            # Need to know hidden size? assumed from default config or saved config?
            # Re-creating model relies on self.cfg which is passed in __init__
            state_dim = self.window * 6
            self.model = PolicyNet(state_dim, self.cfg.hidden_size).to(self.device)
            self.model.load_state_dict(torch.load(path))
            return True
        except Exception:
            return False

# PBT Adapter
class RLModelAdapter(ModelAdapter):
    def __init__(self, device: str = "cpu", window: int = 10):
        self.device = torch.device(device)
        self.window = window
        
    def train_step(self, member: Member, dataset: pd.DataFrame, steps: int) -> Tuple[Any, float]:
        """Train RL for 'steps' episodes and return (new_model_state, avg_reward)."""
        cfg = member.config
        
        state_dim = self.window * 6
        hidden = cfg.get("hidden_size", 128)
        
        model = PolicyNet(state_dim, hidden).to(self.device)
        if member.model_state is not None:
            model.load_state_dict(member.model_state)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate", 1e-3))
        
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = dataset[red_cols].to_numpy(dtype=float) / 33.0
        
        model.train()
        total_reward = 0.0
        running_baseline = 0.0
        entropy_coef = 0.01
        
        for ep in range(steps):
            if len(data) < self.window + 1:
                break
            indices = np.random.randint(self.window, len(data), size=min(32, len(data) - self.window))
            
            log_probs = []
            rewards = []
            entropies = []
            
            for idx in indices:
                state = data[idx - self.window : idx].flatten()
                target_raw = dataset[red_cols].iloc[idx].to_numpy(dtype=int)
                
                state_t = torch.FloatTensor(state).to(self.device)
                probs = model(state_t)
                m = torch.distributions.Bernoulli(probs)
                action = m.sample()
                
                entropy = m.entropy().sum()
                entropies.append(entropy)
                
                selected = torch.nonzero(action).squeeze(-1).cpu().numpy() + 1
                matches = len(set(selected) & set(target_raw))
                cost = len(selected)
                
                reward = matches * 10.0 - cost * 1.0
                if matches >= 3:
                    reward += 20.0
                
                log_probs.append(m.log_prob(action).sum())
                rewards.append(reward)
                total_reward += reward
            
            if not rewards:
                continue
                
            avg_r = np.mean(rewards)
            running_baseline = 0.1 * avg_r + 0.9 * running_baseline
            advantages = [r - running_baseline for r in rewards]
            
            optimizer.zero_grad()
            policy_loss = torch.stack([-lp * adv for lp, adv in zip(log_probs, advantages)]).mean()
            entropy_bonus = -entropy_coef * torch.stack(entropies).mean()
            loss = policy_loss + entropy_bonus
            loss.backward()
            optimizer.step()
        
        avg_reward = total_reward / max(steps, 1)
        return model.state_dict(), avg_reward
    
    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        """Evaluate by simulating predictions on last 10% of data."""
        cfg = member.config
        state_dim = self.window * 6
        model = PolicyNet(state_dim, cfg.get("hidden_size", 128)).to(self.device)
        if member.model_state is not None:
            model.load_state_dict(member.model_state)
        
        model.eval()
        red_cols = ["red1", "red2", "red3", "red4", "red5", "red6"]
        data = dataset[red_cols].to_numpy(dtype=float) / 33.0
        
        n_test = max(1, len(data) // 10)
        hits = 0
        
        with torch.no_grad():
            for idx in range(len(data) - n_test, len(data)):
                if idx < self.window:
                    continue
                state = data[idx - self.window : idx].flatten()
                state_t = torch.FloatTensor(state).to(self.device)
                probs = model(state_t).cpu().numpy()
                top6 = np.argsort(probs)[::-1][:6] + 1
                target = dataset[red_cols].iloc[idx].to_numpy(dtype=int)
                hits += len(set(top6) & set(target))
        
        return hits / (n_test * 6)
    
    def perturb_config(self, config: Any) -> Any:
        """Randomly perturb hyperparameters."""
        new_config = config.copy()
        new_config["hidden_size"] = max(32, config.get("hidden_size", 128) + np.random.randint(-32, 33))
        new_config["learning_rate"] = np.clip(config.get("learning_rate", 1e-3) * np.random.choice([0.8, 1.0, 1.25]), 1e-5, 0.1)
        return new_config
    
    def save(self, member: Member, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(member.model_state, path / "rl_pbt.pt")
    
    def load(self, path: Path) -> Member:
        state = torch.load(path / "rl_pbt.pt", weights_only=True)
        return Member(config={}, model_state=state)

