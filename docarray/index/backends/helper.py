from typing import Any, Dict, List, Tuple, Type, cast, Set

from docarray import BaseDoc, DocList
from docarray.index.abstract import BaseDocIndex
from docarray.utils.filter import filter_docs
from docarray.utils.find import FindResult


def _collect_query_args(method_name: str):  # TODO: use partialmethod instead

    return inner


def _collect_query_required_args(method_name: str, required_args: Set[str] = None):
    """
    Returns a function that ensures required keyword arguments are provided.

    :param method_name: The name of the method for which the required arguments are being checked.
    :type method_name: str
    :param required_args: A set containing the names of required keyword arguments. Defaults to None.
    :type required_args: Optional[Set[str]]
    :return: A function that checks for required keyword arguments before executing the specified method.
        Raises ValueError if positional arguments are provided.
        Raises TypeError if any required keyword argument is missing.
    :rtype: Callable
    """

    if required_args is None:
        required_args = set()


    return inner


def _execute_find_and_filter_query(
    doc_index: BaseDocIndex, query: List[Tuple[str, Dict]], reverse_order: bool = False
) -> FindResult:
    """
    Executes all find calls from query first using `doc_index.find()`,
    and filtering queries after that using DocArray's `filter_docs()`.

    Text search is not supported.

    :param doc_index: Document index instance.
        Either InMemoryExactNNIndex or HnswDocumentIndex.
    :param query: Dictionary containing search and filtering configuration.
    :param reverse_order: Flag indicating whether to sort in descending order.
        If set to False (default), the sorting will be in ascending order.
        This option is necessary because, depending on the index, lower scores
        can correspond to better matches, and vice versa.
    :return: Sorted documents and their corresponding scores.
    """
    pass
