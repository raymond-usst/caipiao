
import logging
import sys
import os

# Improved Filter setup
class SeedFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Seed set to" in msg:
            return False
        return True

seed_filter = SeedFilter()
# logging.getLogger().addFilter(seed_filter) # Root filter might not be enough if propagate is false

names = [
    "lightning", 
    "pytorch_lightning", 
    "lightning.fabric", 
    "lightning.pytorch",
    "lightning.fabric.utilities.seed",
    "pytorch_lightning.utilities.seed"
]

for name in names:
    logger = logging.getLogger(name)
    logger.setLevel(logging.ERROR)
    logger.addFilter(seed_filter)
    logger.propagate = False # Prevent propagation to root if root prints it. 
    # Actually, allow propagation but filter?
    # Usually seed_everything might add a handler to the logger. We need to clear it or filter it.

# Also, directly filter the handlers if any
for name in names:
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        handler.addFilter(seed_filter)

# Simulate
try:
    import pytorch_lightning as pl
    print("Calling pl.seed_everything(42)")
    pl.seed_everything(42)
except ImportError:
    print("pytorch_lightning not found")
