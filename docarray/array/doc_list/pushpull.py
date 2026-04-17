import logging
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    Iterator,
    Tuple,
    Type,
    TypeVar,
    cast,
)

from typing_extensions import Literal
from typing_inspect import get_args

PUSH_PULL_PROTOCOL = Literal['s3', 'file']
SUPPORTED_PUSH_PULL_PROTOCOLS = get_args(PUSH_PULL_PROTOCOL)

if TYPE_CHECKING:  # pragma: no cover
    from docarray import BaseDoc, DocList
    from docarray.store.abstract_doc_store import AbstractDocStore


SelfPushPullMixin = TypeVar('SelfPushPullMixin', bound='PushPullMixin')


class PushPullMixin(Iterable['BaseDoc']):
    """Mixin class for push/pull functionality."""

    __backends__: Dict[str, Type['AbstractDocStore']] = {}
    doc_type: Type['BaseDoc']

    @abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    def resolve_url(url: str) -> Tuple[PUSH_PULL_PROTOCOL, str]:
        """Resolve the URL to the correct protocol and name.
        :param url: url to resolve
        """
        pass

    @classmethod
    def get_pushpull_backend(
        cls: Type[SelfPushPullMixin], protocol: PUSH_PULL_PROTOCOL
    ) -> Type['AbstractDocStore']:
        """
        Get the backend for the given protocol.

        :param protocol: the protocol to use, e.g. 'file', 's3'
        :return: the backend class
        """
        pass

    def push(
        self,
        url: str,
        show_progress: bool = False,
        **kwargs,
    ) -> Dict:
        """Push this `DocList` object to the specified url.

        :param url: url specifying the protocol and save name of the `DocList`. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: If true, a progress bar will be displayed.
        """
        pass

    @classmethod
    def push_stream(
        cls: Type[SelfPushPullMixin],
        docs: Iterator['BaseDoc'],
        url: str,
        show_progress: bool = False,
    ) -> Dict:
        """Push a stream of documents to the specified url.

        :param docs: a stream of documents
        :param url: url specifying the protocol and save name of the `DocList`. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: If true, a progress bar will be displayed.
        """
        pass

    @classmethod
    def pull(
        cls: Type[SelfPushPullMixin],
        url: str,
        show_progress: bool = False,
        local_cache: bool = True,
    ) -> 'DocList':
        """Pull a `DocList` from the specified url.

        :param url: url specifying the protocol and save name of the `DocList`. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded `DocList` to local folder
        :return: a `DocList` object
        """
        pass

    @classmethod
    def pull_stream(
        cls: Type[SelfPushPullMixin],
        url: str,
        show_progress: bool = False,
        local_cache: bool = False,
    ) -> Iterator['BaseDoc']:
        """Pull a stream of Documents from the specified url.

        :param url: url specifying the protocol and save name of the `DocList`. Should be of the form ``protocol://namespace/name``. e.g. ``s3://bucket/path/to/namespace/name``, ``file:///path/to/folder/name``
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded `DocList` to local folder
        :return: Iterator of Documents
        """
        pass
