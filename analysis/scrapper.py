import logging
import os
import pandas as pd
import csv 

home_path = os.path.expanduser('~')
root_path = os.path.abspath('./')
os.makedirs(home_path + "/" + "logger", exist_ok=True)
logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename= home_path + '/logger/analysis.log',
                    filemode='a')

logger = logging.getLogger(__name__)


def get_median_from_csv(file_path, metric_column='Metric', value_column='Value', median_metric='median'):
    try:
        data = pd.read_csv(file_path)
        median_value = data[data[metric_column] == median_metric][value_column].iloc[0]
        return median_value
    except Exception as e:
        logger.info(f"Error occurred: {e}")
        return None
def write_2d_list_to_csv(data_2d, file_path):
    try:
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_2d)
        logger.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logger.info(f"Error occurred while writing to the file: {e}")
        
rows = [['model_idx', 'tf_time', 'lite_time']]
for i in range(1, 13825):
    tf_inference_time = get_median_from_csv(home_path + f"/inference_time/{i}/tf_time.csv")
    lite_inference_time = get_median_from_csv(home_path + f"/inference_time/{i}/lite_time.csv")
    row = [
        i, 
        tf_inference_time, 
        lite_inference_time
    ]
    rows.append(row)
    logger.info(f"{i} --> DONE")

write_2d_list_to_csv(rows, home_path.split("/")[2] + "_" + "data.csv")


    
    





