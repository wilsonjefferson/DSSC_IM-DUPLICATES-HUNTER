"""
    This script contains the FileHandler class to handle
    a (excel) file.

    @file: CompareCodes.py
    @version: v. 1.2.1
    @last update: 23/06/2022
    @author: Pietro Morichetti
"""

import re


class CompareCodes:
    '''
        This is a support class, it can be used to compare two tickets code.

        Attributes
        ----------
        None
    '''

    def split(self, code:str) -> tuple:
        '''
            Split a ticket code in ticket category and digit parts.

            Parameters
            ----------
            code: str
                Ticket code

            Returns
            -------
            tuple
                A tuple composed by ticket type and digit part
        '''

        match = re.match(r"([a-z]+)(\d+)", code, re.I)
        return match.groups()
    
    def greater(self, code1:str, code2:str) -> bool:
        '''
            Check if a ticket is greater to another ticket

            Parameters
            ----------
            code1: str
                Ticket code

            code2: str
                Ticket code

            Returns
            -------
            Bool
                True if the code1 is greater than code2, False otherwise
        '''

        text1, digit1 = self.split(code1)
        text2, digit2 = self.split(code2)
        digit1 = int(digit1)
        digit2 = int(digit2)

        if text1 == text2 and digit1 > digit2:
            return True
        return False
