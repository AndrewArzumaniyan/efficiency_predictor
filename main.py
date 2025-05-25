import logging
import logging.config
import os
import datetime

def setup_logging():
    log_dir = 'logs'
    log_filename = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Application started")


if __name__ == "__main__":
    main()