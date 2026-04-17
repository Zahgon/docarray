# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Type, TypeVar

from typing_extensions import TYPE_CHECKING

from docarray.store.abstract_doc_store import AbstractDocStore
from docarray.store.exceptions import ConcurrentPushException
from docarray.store.helpers import _from_binary_stream, _to_binary_stream
from docarray.utils._internal.cache import _get_cache_path

if TYPE_CHECKING:
    from docarray import BaseDoc, DocList

SelfFileDocStore = TypeVar('SelfFileDocStore', bound='FileDocStore')


class FileDocStore(AbstractDocStore):
    """Class to push and pull [`DocList`][docarray.DocList] on-disk."""

    @staticmethod
    def _abs_filepath(name: str) -> Path:
        """Resolve a name to an absolute path.

        :param name: If it is not a path, the cache directory is prepended.
            If it is a path, it is resolved to an absolute path.
        :return: Path
        """
        pass

    @classmethod
    def list(
        cls: Type[SelfFileDocStore], namespace: str, show_table: bool
    ) -> List[str]:
        """List all [`DocList`s][docarray.DocList] in a directory.

        :param namespace: The directory to list.
        :param show_table: If True, print a table of the files in the directory.
        :return: A list of the names of the `DocLists` in the directory.
        """
        pass

    @classmethod
    def delete(
        cls: Type[SelfFileDocStore], name: str, missing_ok: bool = False
    ) -> bool:
        """Delete a [`DocList`][docarray.DocList] from the local filesystem.

        :param name: The name of the `DocList` to delete.
        :param missing_ok: If True, do not raise an exception if the file does not exist. Defaults to False.
        :return: True if the file was deleted, False if it did not exist.
        """
        pass

    @classmethod
    def push(
        cls: Type[SelfFileDocStore],
        docs: 'DocList',
        name: str,
        show_progress: bool,
    ) -> Dict:
        """Push this [`DocList`][docarray.DocList] object to the specified file path.

        :param docs: The `DocList` to push.
        :param name: The file path to push to.
        :param show_progress: If true, a progress bar will be displayed.
        """
        pass

    @classmethod
    def push_stream(
        cls: Type[SelfFileDocStore],
        docs: Iterator['BaseDoc'],
        name: str,
        show_progress: bool = False,
    ) -> Dict:
        """Push a stream of documents to the specified file path.

        :param docs: a stream of documents
        :param name: The file path to push to.
        :param show_progress: If true, a progress bar will be displayed.
        """
        pass

    @classmethod
    def pull(
        cls: Type[SelfFileDocStore],
        docs_cls: Type['DocList'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> 'DocList':
        """Pull a [`DocList`][docarray.DocList] from the specified url.

        :param name: The file path to pull from.
        :param show_progress: if true, display a progress bar.
        :param local_cache: store the downloaded `DocList` to local folder
        :return: a `DocList` object
        """
        pass

    @classmethod
    def pull_stream(
        cls: Type[SelfFileDocStore],
        docs_cls: Type['DocList'],
        name: str,
        show_progress: bool,
        local_cache: bool,
    ) -> Iterator['BaseDoc']:
        """Pull a stream of Documents from the specified file.

        :param name: The file path to pull from.
        :param show_progress: if true, display a progress bar.
        :param local_cache: Not used by the ``file`` protocol.
        :return: Iterator of Documents
        """
        pass
