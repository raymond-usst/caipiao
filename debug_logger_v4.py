
import logging
import pytorch_lightning as pl

print(f"PL Version: {pl.__version__}")

# Check loggers AFTER import
print("\nLoggers:")
for name in logging.root.manager.loggerDict:
    if "seed" in name or "light" in name:
        print(name)

# Try identifying the logger by capturing the record
class CaptureFilter(logging.Filter):
    def filter(self, record):
        if "Seed set to" in record.getMessage():
            print(f"\n[CAPTURED] Logger Name: {record.name}")
            print(f"[CAPTURED] Level: {record.levelno}")
            print(f"[CAPTURED] Path: {record.pathname}")
            return False # Suppress it to prove we caught it
        return True

capture = CaptureFilter()
logging.getLogger().addFilter(capture)
# Add to root handlers too
logging.basicConfig()
for h in logging.getLogger().handlers:
    h.addFilter(capture)

print("\nCalling pl.seed_everything(42)")
pl.seed_everything(42)
