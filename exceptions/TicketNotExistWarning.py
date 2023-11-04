from exceptions.TicketWarning import TicketWarning


class TicketNotExistWarning(TicketWarning):
    """
        Warning in case a not existing header is given.
        
        Parameters
        ----------
        header: str
            an header of a table
    """

    def __init__(self, ticket_code: str):
        self.message = "ATTENTION: ticket {} not in Excel file".format(ticket_code)
        super().__init__(self.message)
