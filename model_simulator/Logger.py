import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "my_app.log"
path = os.path.dirname(os.path.abspath(__file__))
logging.config.fileConfig(path + '\\logging.ini')
fileh = logging.FileHandler(LOG_FILE, 'a')
fileh.setFormatter(FORMATTER)


def get_logger(logger_name):
   logger = logging.getLogger(logger_name)
   logger.setLevel(logging.DEBUG) # better to have too much log than not enough

   # with this pattern, it's rarely necessary to propagate the error up to parent
   logger.propagate = False
   return logger