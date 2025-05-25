import logging
import logging.config
import os
from datetime import datetime

from src.utils import setup_logging


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Application started")


if __name__ == "__main__":
    main()