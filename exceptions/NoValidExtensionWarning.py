from exceptions.InputHandlerWarning import InputHandlerWarning


class NoValidExtensionWarning(InputHandlerWarning):
    """
        Warning in case a not valid extension of the file is given absolute file path is given.
        The given path might be a database location, that's why a Warning class is used.

        Parameters
        ----------
        None
    """
    def __init__(self, path_file: str):
        self.message = "ATTENTION: file {} has a no valid extension!".format(path_file)
        super().__init__(self.message)
