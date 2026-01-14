
import logging
import pytorch_lightning as pl

# Print all log records
class PrintFilter(logging.Filter):
    def filter(self, record):
        print(f"Log: '{record.getMessage()}' Name: {record.name}")
        return True

logging.getLogger().addFilter(PrintFilter())
logging.basicConfig(level=logging.INFO)
for h in logging.getLogger().handlers:
    h.addFilter(PrintFilter())

print("Calling pl.seed_everything(42)")
pl.seed_everything(42)
