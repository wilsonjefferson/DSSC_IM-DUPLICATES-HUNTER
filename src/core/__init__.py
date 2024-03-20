import logging
import logging.config
from os import path, listdir, makedirs
from configparser import ConfigParser

from ConfigNameSpace import MAIN_STAGE, MAIN_TRAIN


# After the backup_location is defined in ConfigNameSpace, we can proceed by
# modifying the content of log.config (but not the file itself!). This file 
# contain the information to create a configur the custom loggers and for those
# using a FileHandler a destination file where to store the logs is required.
# That is why log.config content is "adjusted", before the loggers are instantiated.

backup_location = MAIN_STAGE.backup_location
backup_logs_location = path.join(backup_location, 'logs')


# create the backup folders if not exist
makedirs(backup_location, exist_ok=True)
makedirs(backup_location + 'plots', exist_ok=True)
makedirs(backup_location + 'logs', exist_ok=True)
makedirs(backup_location + 'pred_probs', exist_ok=True)
makedirs(backup_location + 'step_compute_similarities', exist_ok=True)

if MAIN_STAGE is MAIN_TRAIN:
    makedirs(backup_location + 'step_construct_duplicates_dict', exist_ok=True)
    makedirs(backup_location + 'step_text_preprocessing/text_filtering_chuncks/text', exist_ok=True)
    makedirs(backup_location + 'tfidf', exist_ok=True)
    makedirs(backup_location + 'fasttext', exist_ok=True)
    makedirs(backup_location + 'doc2vec', exist_ok=True)
    makedirs(backup_location + 'step_construct_dataset_original/datasetbuilder_tmp_files', exist_ok=True)
    makedirs(backup_location + 'step_construct_dataset_artificial', exist_ok=True)
    makedirs(backup_location + 'models', exist_ok=True)
else:
    makedirs(backup_location + 'step_reduce_search_space', exist_ok=True)
    makedirs(backup_location + 'step_text_preprocessing', exist_ok=True)
    makedirs(backup_location + 'step_assess_on_new_tickets', exist_ok=True)

run_number = [name for name in listdir(backup_logs_location) 
              if path.isfile(path.join(backup_logs_location, name))]
run_number = len(run_number) + 1

location = path.join(backup_logs_location, f'logs_run_{run_number}.log')
logging.info(f'Log file: {location}')

curdir = path.dirname(path.abspath(__file__))
log_file_path = path.join(curdir, 'log.config')

file_content = None
with open(log_file_path, 'r') as file:
    file_content = file.read()

# {LOCATION} keyword in log.config to be replaced by a proper location
file_content = file_content.replace('{LOCATION}', location)

# create ConfigParser to read the adjusted log.config file in string format
config = ConfigParser()
config.read_string(file_content)

logging.config.fileConfig(fname=config, disable_existing_loggers=False)
