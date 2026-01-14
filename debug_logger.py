
import logging
import pytorch_lightning as pl

# Print all loggers
print("Loggers before seed_everything:")
for name in logging.root.manager.loggerDict:
    print(name)

print("\n--- Calling seed_everything(42) ---")
pl.seed_everything(42)

print("\nLoggers after seed_everything:")
for name in logging.root.manager.loggerDict:
    print(name)

# Check specifically for "lightning.fabric.utilities.seed"
log = logging.getLogger("lightning.fabric.utilities.seed")
print(f"\nLogger: lightning.fabric.utilities.seed")
print(f"Level: {log.level}")
print(f"Handlers: {log.handlers}")
print(f"Propagate: {log.propagate}")
print(f"Parent: {log.parent.name if log.parent else 'None'}")
