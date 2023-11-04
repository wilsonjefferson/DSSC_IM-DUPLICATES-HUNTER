from exceptions.InputHandlerException import InputHandlerException


class SourceNotValidException(InputHandlerException):
    """
        Exception in case the source is not an excel file or a db table.

        Parameters
        ----------
        source : exception message
    """

    def __init__(self, source: str):
        self.message = "ATTENTION: {} is not a valid source!".format(source)
        super().__init__(self.message)
