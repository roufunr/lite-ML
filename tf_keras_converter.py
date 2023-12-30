from tensorflow.keras.models import load_model

import logging
import sys
import os

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')

logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/keras_converter.log',
                    filemode='a')
logger = logging.getLogger(__name__)

for i in range(1, 13824 + 1):
    model_path = f"/home/rouf-linux/lite-ML/models/tf/{i}"
    loaded_model = load_model(model_path)
    loaded_model.save(f"/home/rouf-linux/lite-ML/models/tf_keras/{i}.keras")
    logger.info(f"{i} -- DONE")