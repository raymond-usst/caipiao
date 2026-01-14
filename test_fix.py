
import logging
import sys
import os

# Proposed Fix in main.py
class SeedFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Seed set to" in msg:
            return False
        return True

seed_filter = SeedFilter()

# 1. Add filter to root handlers (if any exist now or later?) 
# Note: Handlers might be added later by libraries. 
# So targeting the specific logger is better.

# Specific loggers that emit "Seed set to"
seed_loggers = [
    "lightning.fabric.utilities.seed",
    "pytorch_lightning.utilities.seed"
]

for name in seed_loggers:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR) # Mute INFO
    logger.addFilter(seed_filter)  # And filter just in case

# Simulate
try:
    import pytorch_lightning as pl
    print("Calling pl.seed_everything(42)")
    pl.seed_everything(42)
except ImportError:
    print("pytorch_lightning not found")
