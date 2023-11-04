from exceptions.InputHandlerWarning import InputHandlerWarning


class HeaderNotExistWarning(InputHandlerWarning):
    """
        Warning in case a not existing header is given.
        
        Parameters
        ----------
        header: str
            an header of a table
    """
    def __init__(self, header: str):
        self.message = "ATTENTION: header {} does not exist!".format(header)
        super().__init__(self.message)
