from exceptions.RDTException import RDTException


class InputHandlerException(RDTException):
    """
        General Exception for the FileHandler operations.
        
        Parameters
        ----------
        message : exception message
    """
    def __init__(self, message: str):
        super().__init__(message)
