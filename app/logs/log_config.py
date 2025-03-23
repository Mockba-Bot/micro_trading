import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import os

class DateRotatingFileHandler(RotatingFileHandler):
    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Create a new log file name with the current date and time
        dfn = self.baseFilename + datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        
        # Ensure the new log file name does not already exist
        if os.path.exists(dfn):
            os.remove(dfn)
        
        # Rotate the files
        self.rotate(self.baseFilename, dfn)
        
        # Open a new log file
        if not self.delay:
            self.stream = self._open()

# Configure trader logger
trader_logger = logging.getLogger('trader_logger')
if not trader_logger.hasHandlers():
    trader_logger.setLevel(logging.DEBUG)

    trader_handler = DateRotatingFileHandler('logs/trader.log', maxBytes=5*1024*1024, backupCount=0)
    trader_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    trader_handler.setFormatter(formatter)
    trader_logger.addHandler(trader_handler)

# Configure gainers logger
gainers_logger = logging.getLogger('gainers_logger')
if not gainers_logger.hasHandlers():
    gainers_logger.setLevel(logging.DEBUG)

    gainers_handler = DateRotatingFileHandler('logs/gainers.log', maxBytes=5*1024*1024, backupCount=0)
    gainers_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    gainers_handler.setFormatter(formatter)
    gainers_logger.addHandler(gainers_handler)

# Configure gainers logger
signal_logger = logging.getLogger('signal_logger')
if not signal_logger.hasHandlers():
    signal_logger.setLevel(logging.DEBUG)

    signal_handler = DateRotatingFileHandler('logs/signal.log', maxBytes=5*1024*1024, backupCount=0)
    signal_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    signal_handler.setFormatter(formatter)
    signal_logger.addHandler(signal_handler)    


# Configure scalper logger
scalper_logger = logging.getLogger('scalper_logger')
if not scalper_logger.hasHandlers():
    scalper_logger.setLevel(logging.DEBUG)

    scalper_handler = DateRotatingFileHandler('logs/scalper.log', maxBytes=5*1024*1024, backupCount=0)
    scalper_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    scalper_handler.setFormatter(formatter)
    scalper_logger.addHandler(scalper_handler)