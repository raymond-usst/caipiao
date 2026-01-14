
import logging
import sys
import os

class SeedFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Seed set to" in msg:
            return False
        return True

seed_filter = SeedFilter()
name = "lightning.fabric.utilities.seed"

logger = logging.getLogger(name)
logger.setLevel(logging.ERROR)
logger.addFilter(seed_filter)

print(f"Before import: {name} Level={logger.level} Filters={logger.filters}")

import pytorch_lightning as pl
from lightning.fabric.utilities.seed import seed_everything as fabric_seed

logger = logging.getLogger(name)
print(f"After import: {name} Level={logger.level} Filters={logger.filters} Handlers={logger.handlers} Propagate={logger.propagate}")

print("Calling pl.seed_everything(42)")
pl.seed_everything(42)  # This one

print("Calling fabric_seed(43)")
fabric_seed(43)       # This one
