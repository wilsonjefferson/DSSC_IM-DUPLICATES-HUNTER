import os
import yaml
from recordclass import recordclass, asdict
from dataclasses import dataclass
from datetime import datetime, timedelta

CONFIG_PATH = './src/config.yml'
DATE_FORMAT = '%Y-%m-%d'
TODAY = datetime.today().strftime(DATE_FORMAT)
YESTERDAY = (datetime.today() - timedelta(days = 1)).strftime(DATE_FORMAT)


@dataclass
class ConfigNameSpace:

    def __init__(self, config_path:str):
        with open(config_path, 'r') as file:
            dict_yaml = yaml.safe_load(file)
        
        constants, _ = self._read_config(dict_yaml)
        config_vars, config_vals = zip(*constants)
        RecordNamedClass = recordclass('GLOBAL', config_vars)
        self.constants = RecordNamedClass(*config_vals)
    
    def _read_config(self, d:dict):
        keys, vals = list(), list()
        for k, v in d.items():
            if isinstance(v, dict):
                tmp_keys, tmp_vals = self._read_config(v)
                RecordNamedClass = recordclass(k, tmp_keys)
                const = RecordNamedClass(*tmp_vals)
                keys.append([k, const])
            else:
                keys.append(k)
                vals.append(v)
        return keys, vals
    
    def store(self, location:str = None):
        if location is None:
            location = config_backup_path
        
        dict_config = asdict(self._constants)
        dict_yaml = self._todict(dict_config)

        with open(location, 'w') as file:
            yaml.dump(dict_yaml, file)

    def _todict(self, dict_config:dict) -> dict:
        dict_yaml = dict()
        for k, v in dict_config.items():
            if not (isinstance(v, str) or isinstance(v, bool) or \
                    isinstance(v, int) or isinstance(v, datetime)):
                try:
                    v = self._todict(asdict(v))
                except Exception as e:
                    print('ERROR in _todict function - ', k, ': ', type(v), v)
                    raise e
            dict_yaml[k] = v
        return dict_yaml

    @property
    def constants(self):
        return self._constants
    
    @constants.setter
    def constants(self, constants) -> None:
        self._constants = constants

    def __repr__(self):
        dict_config = asdict(self._constants)
        dict_yaml = self._todict(dict_config)

        sto = ''
        for k, v in dict_yaml.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    sto += f'    {kk}: {vv}\n'
            else:
                sto += f'    {k}: {v}\n'
        return sto


def exist_config_backup():
    
    backups_path = [os.path.join(_global.backup_root, f) \
                        for f in os.listdir(_global.backup_root)]
    backups_path = sorted(backups_path, key=os.path.getmtime, reverse=True)

    if len(backups_path) == 0:
        from copy import deepcopy

        print('No backup found, proceding with default ConfigNameSpace')
        _config_name_space = deepcopy(_default_config_name_space)
        return None, _config_name_space

    backup_path = backups_path[0] # more recent backup
    config_backup_path = backup_path + '/config.yml'
    _config_name_space = ConfigNameSpace(config_path=config_backup_path)
    print(f'Backup found, proceeding with backup from {config_backup_path}')
    
    return config_backup_path, _config_name_space


# define ConfigNameSpace with default configuration file
_default_config_name_space = ConfigNameSpace(CONFIG_PATH)
_config_vars = _default_config_name_space.constants
_global = _config_vars.GLOBAL

# define backup and package root base on the engine system: local or Azure
if os.getenv("AML_CloudName") == 'AzureCloud':
    backup_root = _global.backup_root
    package_root = _global.package_root
else:
    raise Exception('ConfigNameSpace: Unknown or missing configuration for current base system')

# determine if exist a custom configuration file
config_backup_path, _config_name_space = exist_config_backup()
CONFIG_VARS = _config_name_space.constants
GLOBAL = CONFIG_VARS.GLOBAL

# define universal variables
last_time = GLOBAL.last_incident_time
project_root = GLOBAL.project_root

MAIN_TRAIN = CONFIG_VARS.MAIN_TRAIN
MAIN_BUILD = CONFIG_VARS.MAIN_BUILD
MAIN_DEPLOY = CONFIG_VARS.MAIN_DEPLOY
MAIN_STAGE = None

if config_backup_path is None: # if no custom configuration file

    MAIN_TRAIN.backup_location = backup_root + MAIN_TRAIN.backup_location%TODAY
    MAIN_BUILD.backup_location = backup_root + MAIN_BUILD.backup_location%TODAY
    MAIN_DEPLOY.backup_location = backup_root + MAIN_DEPLOY.backup_location%TODAY

    GLOBAL.project_root = os.path.dirname(os.path.realpath(__file__))
    if os.getenv("AML_CloudName") == 'AzureCloud':
        import re
        azure_username = re.findall('Users/(.*)/filtering-of-recognisable-duplicate-tickets', GLOBAL.project_root)[0]
        GLOBAL.package_root = GLOBAL.package_root%azure_username
    
    config_location = f'backup_{TODAY}'

    from os import makedirs
    makedirs(backup_root + config_location)
    _config_name_space.store(backup_root + config_location + '/config.yml')

    # _default_config_name_space.constants.GLOBAL.config_location = GLOBAL.config_location
    _default_config_name_space.store(CONFIG_PATH)

# define universal variable package_root
package_root = GLOBAL.package_root

# set environment variables
os.environ["HTTP_PROXY"] = "http://10.141.0.165:3128"
os.environ["HTTPS_PROXY"] = "http://10.141.0.165:3128"
os.environ["NO_PROXY"] = "localhost, 127.0.0.1, *.azureml.net"
os.environ["NLTK_DATA"] = package_root+'/nltk_data/'
os.environ["TOKENIZERS_PARALLELISM"] = 'true'
