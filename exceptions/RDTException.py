class RDTException(Exception):
    """
        General Exception class of the project.
        Parameters
        ----------
        message : exception message
    """

    def __init__(self, message: str):
        super().__init__(message)
