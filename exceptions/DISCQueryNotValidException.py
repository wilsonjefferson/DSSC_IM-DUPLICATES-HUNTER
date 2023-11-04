from core.data_manipulation.data_handlers.QueryHandler import QueryHandler
from dataclasses import fields

from exceptions.InputHandlerException import InputHandlerException


class DISCQueryNotValidException(InputHandlerException):
    """
        Exception in case user provide a no valid query command.

        Parameters
        ----------
        user_query : query provided by the user
        list_query_available_commands: list of currently available query comnmands
    """

    def __init__(self, user_query: str, Query:QueryHandler):

        commds = list(field.name for field in fields(Query))
        self.message = "ATTENTION: %s is not a valid query!\n"%user_query
        self.message = "INFO: Available query command: %s" % ', '.join(map(str, commds))
        super().__init__(self.message)
