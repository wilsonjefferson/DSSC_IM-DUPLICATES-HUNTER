from exceptions.RTDWarning import RTDWarning


class InputHandlerWarning(RTDWarning):
    """
        General Warning for the FileHandler operations.
        
        Parameters
        ----------
        message : exception message
    """
    def __init__(self, message: str):
        super().__init__(message)
