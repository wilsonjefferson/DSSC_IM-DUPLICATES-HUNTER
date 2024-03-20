from exceptions.InputHandlerWarning import InputHandlerWarning


class FileNotExistWarning(InputHandlerWarning):
    """
        Warning in case a not existing absolute file path is given.
        The given path might be a database location, that's why a Warning class is used.

        Parameters
        ----------
        None
    """
    def __init__(self, path_file: str):
        self.message = "ATTENTION: {} does not exist or is not a file.".format(path_file)
        super().__init__(self.message)
