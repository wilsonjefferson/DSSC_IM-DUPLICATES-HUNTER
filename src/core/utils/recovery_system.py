import pandas as pd
import pickle
import logging
import joblib
from functools import wraps
from typing import Callable, List

from exceptions.NoValidExtensionWarning import NoValidExtensionWarning
from exceptions.RDTException import RDTException

log = logging.getLogger('recoverySystemLogger')

from ConfigNameSpace import MAIN_STAGE
backup_location = MAIN_STAGE.backup_location

import core


def recovery_decorator(backup_locations:List[str]):
    '''
        This decorator represent the recovery system of the project.its aim is to
        store partial results of the train and build-catalog workflow, this becuase
        these workflow may take long time to complete and it is important to save 
        intermidiate results in case of unexpected failures in the code or miss-bihaviour.

        The recovery system create a new folder and store inside it all the partial
        results, in case the code would be interrupted, the next execution will retrieve back
        the partial results in short time and the program will continue from the last
        step of the workflow before the interruption.

        Parameters
        ----------
        backup_locations: list
            List locations for the backup

        Returns
        -------
        inner: object
            Inner decorator
    '''
    def inner(func:Callable):
        '''
            Inner decorator.

            Parameters
            ----------
            func: object
                Function of a workflow to be executed

            Returns
            -------
            wrapper: object
                Inner decorator
        '''
        @wraps(func)
        def wrapper(*args, **kwargs):
            '''
                Inner decorator.

                Parameters
                ----------
                args: list
                    List of argument for the deploy components function
                
                kwargs: dictionary
                    Dictionary of argument for the deploy components function

                Returns
                -------
                Return of function results
            '''
            backups = list(map(_restore, backup_locations)) # executing restore from locations
            is_backups_not_valid = any(item is None for item in backups) # workflow is incompleted
            if is_backups_not_valid:
                objs = func(*args, **kwargs)
                tmp_objs = list(objs) if type(objs) == tuple else [objs]
                collection = tuple(zip(tmp_objs, backup_locations))
                # Assumption: order of locations match the order of 
                # the objects returned by the function-component
                storage = list(map(lambda x: _backup(*x), collection)) # executing backup
                if False in storage:
                    raise RDTException('Something went wrong during the storing...')
                return objs # return component results interrupting the loop  
            return backups[0] if len(backups) == 1 else backups
        return wrapper
    return inner

def _restore(location:str) -> object:
    '''
        This function restore the intermidiate result from a given location.

        Parameters
        ----------
        location: str
            Relative location for a certain patial data

        Returns
        -------
        obj: object
            Partial result
    '''
    from pathlib import Path

    location = backup_location + location
    loc = Path(location)
    log.info('restoring object with absolute path: %s'%loc)

    if not loc.is_file():
        log.error(f'file {location} does not exist')
        return None
    
    suffix = loc.suffix

    try:
        if suffix == '.xlsx':
            obj = pd.read_excel(io=location, engine='openpyxl')
        elif suffix == '.pickle':
            with open(location, 'rb') as handle:
                obj = pickle.load(handle)
        elif suffix == '.joblib':
            obj = joblib.load(location)
        else:
            log.error('%s file extension is not supported'%suffix)
            NoValidExtensionWarning(location)
    except Exception as e:
        log.info('%s file error: %s'%(location, e))
        return None
    else:
        log.info('file %s restored successfully'%location)
        return obj

def _backup(obj:object, location:str) -> bool:
    '''
        This function store the intermidiate result in a given location.

        Parameters
        ----------
        obj: object
            Object to be stored
        
        location: str
            Relative location for a certain patial data

        Returns
        -------
            bool
            True if backup was succesfull
    '''

    location = backup_location + location
    if type(obj) == pd.DataFrame:
        obj.to_excel(location, index=False, engine='openpyxl')
    elif type(obj) == dict or type(obj) == list:
        with open(location, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(location, 'wb') as handle:
            joblib.dump(obj, handle)
    return True
