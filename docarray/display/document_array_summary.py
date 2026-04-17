from typing import TYPE_CHECKING, List

from docarray.typing.tensor.abstract_tensor import AbstractTensor

if TYPE_CHECKING:
    from docarray.array import DocVec
    from docarray.array.any_array import AnyDocArray


class DocArraySummary:
    def __init__(self, docs: 'AnyDocArray'):
        self.docs = docs

    def summary(self) -> None:
        """
        Print a summary of this DocList object and a summary of the schema of its
        Document type.
        """
        pass

    @staticmethod
    def _get_stacked_fields(docs: 'DocVec') -> List[str]:  # TODO this might
        # broken
        """
        Return a list of the field names of a DocVec instance that are
        doc_vec, i.e. all the fields that are of type AbstractTensor. Nested field
        paths are separated by dot, such as: 'attr.nested_attr'.
        """
        pass
