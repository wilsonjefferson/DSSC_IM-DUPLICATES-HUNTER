"""
    Class to handle output messages for the
    end-users.

    @file: OutputHandler.py
    @version: v. 1.1
    @last update: 23/05/2022
    @author: Pietro Morichetti
"""

from dataclasses import dataclass


class OutputHandler:

    @dataclass(frozen=True)
    class Message:
        VALID_ITEM: str = '{} is valid.'
        LIST_ITEM: str = 'List of items: '
        STOP_LIST: str = 'Please insert \'stop\' to interrupt the insertion of items.'
