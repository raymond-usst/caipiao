import copy
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures

import numpy as np
import pandas as pd
import torch

@dataclass
class Member:
    """A member of the PBT population."""
    id: int
    step: int
    config: Any  # Model configuration object
    model_state: Any  # Model weights/state
    performance: float = -float('inf')  # Higher is better
    lineage: List[int] = field(default_factory=list)  # Track ancestry

class ModelAdapter(ABC):
    """Interface for models to participate in PBT."""

    @abstractmethod
    def train_step(self, member: Member, dataset: pd.DataFrame, steps: int) -> Tuple[Any, float]:
        """
        Train the member's model for 'steps' (epochs/iterations).
        Returns: (updated_model_state, latest_loss)
        """
        pass

    @abstractmethod
    def evaluate(self, member: Member, dataset: pd.DataFrame) -> float:
        """
        Evaluate the member's model.
        Returns: performance metric (higher is better).
        """
        pass

    @abstractmethod
    def perturb_config(self, config: Any) -> Any:
        """
        Randomly perturb the configuration (Explore step).
        """
        pass

    @abstractmethod
    def save(self, member: Member, path: Path) -> None:
        """Save member checkpoint."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> Member:
        """Load member checkpoint (if needed for resume)."""
        pass

class PBTRunner:
    def __init__(
        self,
        adapter: ModelAdapter,
        dataset: pd.DataFrame,
        population_size: int = 10,
        generations: int = 10,
        steps_per_gen: int = 1,
        fraction: float = 0.2,
        save_dir: str = "models/pbt",
        n_jobs: int = 1,
    ):
        self.adapter = adapter
        self.dataset = dataset
        self.pop_size = population_size
        self.generations = generations
        self.steps_per_gen = steps_per_gen
        self.fraction = fraction
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.population: List[Member] = []

    def initialize(self, initial_configs: List[Any]):
        if len(initial_configs) != self.pop_size:
            # Replicate or sample if sizes don't match
            initial_configs = [copy.deepcopy(random.choice(initial_configs)) for _ in range(self.pop_size)]
        
        for i, cfg in enumerate(initial_configs):
            # Member starts with step 0 and no model state (will be initialized in first train)
            self.population.append(Member(
                id=i,
                step=0,
                config=cfg,
                model_state=None,  # Adapter should handle None state as "init new model"
                lineage=[i]
            ))

    def run(self):
        print(f"[PBT] Starting PBT with pop_size={self.pop_size}, generations={self.generations}")
        
        for gen in range(1, self.generations + 1):
            start_time = time.time()
            print(f"\n[PBT] Generation {gen}/{self.generations}")
            
            # 1. Train
            print(f"[PBT] Training population for {self.steps_per_gen} steps (n_jobs={self.n_jobs})...")
            
            def _train_single(member: Member):
                return self.adapter.train_step(member, self.dataset, self.steps_per_gen)

            if self.n_jobs > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    # Map returns results in order
                    results = list(executor.map(_train_single, self.population))
                
                # Update members (main thread)
                for member, (updated_state, loss) in zip(self.population, results):
                    member.model_state = updated_state
                    member.step += self.steps_per_gen
            else:
                for member in self.population:
                    updated_state, loss = self.adapter.train_step(
                        member, self.dataset, self.steps_per_gen
                    )
                    member.model_state = updated_state
                    member.step += self.steps_per_gen
            
            # 2. Evaluate
            print("[PBT] Evaluating population...")
            performances = []
            
            def _eval_single(member: Member):
                return self.adapter.evaluate(member, self.dataset)
                
            if self.n_jobs > 1:
                 with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    scores = list(executor.map(_eval_single, self.population))
                 
                 for member, score in zip(self.population, scores):
                     member.performance = score
                     performances.append(score)
            else:
                for member in self.population:
                    score = self.adapter.evaluate(member, self.dataset)
                    member.performance = score
                    performances.append(score)
            
            # Stats
            avg_perf = sum(performances) / len(performances)
            max_perf = max(performances)
            print(f"[PBT] Gen {gen} Stats: Avg={avg_perf:.4f}, Max={max_perf:.4f}")
            
            # 3. Exploit and Explore
            if gen < self.generations:  # Don't evolve after the last generation
                self._exploit_and_explore()
            
            # Checkpoint best
            best_idx = np.argmax([m.performance for m in self.population])
            best_member = self.population[best_idx]
            print(f"[PBT] Best Member ID={best_member.id}, Score={best_member.performance:.4f}")
            self.adapter.save(best_member, self.save_dir / f"gen_{gen}_best.pt")
        
        # Return best member of final generation
        best_idx = np.argmax([m.performance for m in self.population])
        return self.population[best_idx]

    def _exploit_and_explore(self):
        # Sort by performance (descending)
        sorted_pop = sorted(self.population, key=lambda m: m.performance, reverse=True)
        
        # Ensure at least 1 top and 1 bottom member for small populations
        cutoff = int(self.pop_size * self.fraction)
        if cutoff < 1:
            cutoff = 1
        # Also ensure we don't replace everyone (need at least 1 top member)
        if cutoff >= self.pop_size:
            cutoff = self.pop_size // 2
            
        top_members = sorted_pop[:self.pop_size - cutoff] # Top (N-cutoff)
        bottom_members = sorted_pop[-cutoff:] # Bottom cutoff
        
        print(f"[PBT] Exploiting: Replacing bottom {len(bottom_members)} with top {len(top_members)}")
        
        # Determine mapping: bottom_i -> top_j (randomly sample from top)
        for bottom_m in bottom_members:
            parent = random.choice(top_members)
            # Exploit: Copy weights and config
            # Needed deepcopy to avoid shared references
            bottom_m.config = copy.deepcopy(parent.config)
            bottom_m.model_state = copy.deepcopy(parent.model_state) # IMPORTANT: For NN, this copies weights
            bottom_m.lineage.append(parent.id)
            bottom_m.performance = parent.performance
            
            # Explore: Perturb config
            bottom_m.config = self.adapter.perturb_config(bottom_m.config)
            print(f"  -> Member {bottom_m.id} exploited Parent {parent.id} and explored new config.")
