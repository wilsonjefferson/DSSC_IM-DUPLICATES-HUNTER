from exceptions.RTDWarning import RTDWarning


class TicketWarning(RTDWarning):
    """
        General Warning for tickets.
        
        Parameters
        ----------
        message : exception message
    """

    def __init__(self, message: str):
        super().__init__(message)
