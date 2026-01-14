
import logging
import sys

# Define Filter
class SeedFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "Seed set to" in msg:
            return False
        return True

# 1. Initialize logging root handler explicitly
logging.basicConfig(level=logging.INFO)

# 2. Add filter to all root handlers
for handler in logging.getLogger().handlers:
    handler.addFilter(SeedFilter())

print("Handlers:", logging.getLogger().handlers)

# Simulate PL
import pytorch_lightning as pl
print("Calling pl.seed_everything(42)")
pl.seed_everything(42)  
print("Done")
