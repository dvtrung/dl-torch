import logging
import logging.handlers as handlers
import time

logger = logging.getLogger('dl_torch')
logger.setLevel(logging.INFO)
logging.basicConfig()

# Here we define our formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#logHandler = handlers.TimedRotatingFileHandler('timed_app.log', when='M', interval=1, backupCount=2)
#logHandler.setLevel(logging.INFO)
# Here we set our logHandler's formatter
#logHandler.setFormatter(formatter)

#logger.addHandler(logHandler)
