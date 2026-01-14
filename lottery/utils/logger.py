import logging
import sys
from pathlib import Path

def setup_logger(name: str = "lottery", log_file: str = "lottery.log", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        # Ensure directory exists if it's in a subdirectory
        if log_path.parent != Path("."):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Filter out "Seed set to" messages
    class SeedFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            if "Seed set to" in msg:
                return False
            return True
            
    seed_filter = SeedFilter()
    logger.addFilter(seed_filter)
    for handler in logger.handlers:
        handler.addFilter(seed_filter)

    return logger

# Default logger instance
logger = setup_logger()
