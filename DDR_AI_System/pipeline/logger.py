"""
logger.py — Structured logging for the DDR AI pipeline.
Provides consistent, coloured console output and optional file logging.
"""
from __future__ import annotations
import logging
import sys
from datetime import datetime
from pathlib import Path


# ── Colour codes (works on Windows 10+ with ANSI enabled) ─────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
WHITE  = "\033[97m"


LEVEL_STYLES = {
    "DEBUG":    (DIM,    "DBG"),
    "INFO":     (CYAN,   "INF"),
    "WARNING":  (YELLOW, "WRN"),
    "ERROR":    (RED,    "ERR"),
    "CRITICAL": (RED + BOLD, "CRT"),
}


class PipelineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour, abbr = LEVEL_STYLES.get(record.levelname, (WHITE, "???"))
        ts   = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        name = record.name.split(".")[-1][:14].ljust(14)

        msg  = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)

        return f"{DIM}{ts}{RESET} {colour}[{abbr}]{RESET} {DIM}{name}{RESET}  {msg}"


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """
    Return a module-level logger with coloured console + optional file handler.

    Args:
        name:     Usually __name__ of the calling module.
        log_file: If provided, also write to this file path.
    """
    logger = logging.getLogger(name)

    if logger.handlers:          # already configured
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(PipelineFormatter())
    logger.addHandler(ch)

    # ── File handler (optional) ───────────────────────────────────────────────
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ── Default pipeline logger ───────────────────────────────────────────────────
pipeline_logger = get_logger(
    "ddr_pipeline",
    log_file=str(Path(__file__).parent / "output" / "pipeline.log"),
)
