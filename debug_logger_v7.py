
import logging
import pytorch_lightning as pl

# Aggressive suppression
loggers = [
    "lightning", 
    "pytorch_lightning", 
    "lightning.fabric", 
    "lightning.pytorch", 
    "lightning.fabric.utilities.seed",
    "pytorch_lightning.utilities.seed"
]

for name in loggers:
    l = logging.getLogger(name)
    l.setLevel(logging.ERROR)
    l.handlers = [] # potentially dangerous if we want other logs
    l.propagate = False

print("Calling pl.seed_everything(42)")
pl.seed_everything(42)  
print("Done")
