import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Type, TypeVar

from docarray.store.abstract_doc_store import AbstractDocStore
from docarray.store.helpers import _from_binary_stream, _to_binary_stream
from docarray.utils._internal.cache import _get_cache_path
from docarray.utils._internal.misc import import_library

if TYPE_CHECKING:  # pragma: no cover
    import boto3
    import botocore
    from smart_open import open

    from docarray import BaseDoc, DocList
else:
    open = import_library('smart_open', raise_error=True).open
    boto3 = import_library('boto3', raise_error=True)
    botocore = import_library('botocore', raise_error=True)

SelfS3DocStore = TypeVar('SelfS3DocStore', bound='S3DocStore')


class _BufferedCachingReader:
    """A buffered reader that writes to a cache file while reading."""

    def __init__(
        self, iter_bytes: io.BufferedReader, cache_path: Optional['Path'] = None
    ):
        self._data = iter_bytes
        self._cache = None
        if cache_path:
            self._cache_path = cache_path.with_suffix('.tmp')
            self._cache = open(self._cache_path, 'wb')
        self.closed = False

    def read(self, size: Optional[int] = -1) -> bytes:
        bytes = self._data.read(size)
        if self._cache:
            self._cache.write(bytes)
        return bytes

    def close(self):
        if not self.closed and self._cache:
            self._cache_path.rename(self._cache_path.with_suffix('.docs'))
            self._cache.close()


class S3DocStore(AbstractDocStore):
    """Class to push and pull [`DocList`][docarray.DocList] to and from S3."""

    @staticmethod
    def list(namespace: str, show_table: bool = False) -> List[str]:
        """List all [`DocList`s][docarray.DocList] in the specified bucket and namespace.

        :param namespace: The bucket and namespace to list. e.g. my_bucket/my_namespace
        :param show_table: If true, a rich table will be printed to the console.
        :return: A list of `DocList` names.
        """
        pass

    @staticmethod
    def delete(name: str, missing_ok: bool = True) -> bool:
        """Delete the [`DocList`][docarray.DocList] object at the specified bucket and key.

        :param name: The bucket and key to delete. e.g. my_bucket/my_key
        :param missing_ok: If true, no error will be raised if the object does not exist.
        :return: True if the object was deleted, False if it did not exist.
        """
        pass

    @classmethod
    def push(
        cls: Type[SelfS3DocStore],
        docs: 'DocList',
        name: str,
        show_progress: bool = False,
    ) -> Dict:
        """Push this [`DocList`][docarray.DocList] object to the specified bucket and key.

        :param docs: The `DocList` to push.
        :param name: The bucket and key to push to. e.g. my_bucket/my_key
        :param show_progress: If true, a progress bar will be displayed.
        """
        pass

    @staticmethod
    def push_stream(
        docs: Iterator['BaseDoc'],
        name: str,
        show_progress: bool = False,
    ) -> Dict:
        """Push a stream of documents to the specified bucket and key.

        :param docs: a stream of documents
        :param name: The bucket and key to push to. e.g. my_bucket/my_key
        :param show_progress: If true, a progress bar will be displayed.
        """
        pass

    @classmethod
    def pull(
        cls: Type[SelfS3DocStore],
        docs_cls: Type['DocList'],
        name: str,
        show_progress: bool = False,
        local_cache: bool = False,
    ) -> 'DocList':
        """Pull a [`DocList`][docarray.DocList] from the specified bucket and key.

        :param name: The bucket and key to pull from. e.g. my_bucket/my_key
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocList to local cache
        :return: a `DocList` object
        """
        pass

    @classmethod
    def pull_stream(
        cls: Type[SelfS3DocStore],
        docs_cls: Type['DocList'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDoc']:
        """Pull a stream of Documents from the specified name.
        Name is expected to be in the format of bucket/key.

        :param name: The bucket and key to pull from. e.g. my_bucket/my_key
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded DocList to local cache
        :return: An iterator of Documents
        """
        pass
