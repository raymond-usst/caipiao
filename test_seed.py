
import logging
import sys
import os

# Filter setup from main.py
class SeedFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Seed set to" in msg:
            return False
        return True

seed_filter = SeedFilter()
logging.getLogger().addFilter(seed_filter)
for name in ["lightning", "pytorch_lightning", "lightning.fabric", "lightning.pytorch"]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.addFilter(seed_filter)

# Simulate what might be happening
try:
    import pytorch_lightning as pl
    print("Calling pl.seed_everything(42)")
    pl.seed_everything(42)
except ImportError:
    print("pytorch_lightning not found")

try:
    from lightning.fabric.utilities.seed import seed_everything
    print("Calling fabric seed_everything(42)")
    seed_everything(42)
except ImportError:
    print("lightning.fabric not found")
