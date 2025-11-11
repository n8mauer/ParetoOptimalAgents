from __future__ import annotations
import logging, os, sys, structlog

def _configure_logging(level: str = "INFO"):
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    pre_chain = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *pre_chain,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO),
    )

def get_logger(name: str = "economic_marl", level: str = None):
    _configure_logging(level)
    return structlog.get_logger(name)
