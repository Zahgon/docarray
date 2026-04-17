import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from docarray import BaseDoc, DocList
from docarray.array.any_array import AnyDocArray
from docarray.helper import _shallow_copy_doc
from docarray.index.abstract import BaseDocIndex, _raise_not_supported
from docarray.index.backends.helper import _collect_query_args
from docarray.typing import AnyTensor, NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils.filter import filter_docs
from docarray.utils.find import (
    FindResult,
    FindResultBatched,
    _extract_embeddings,
    _FindResult,
    _FindResultBatched,
    find,
    find_batched,
)

TSchema = TypeVar('TSchema', bound=BaseDoc)


class InMemoryExactNNIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(
        self,
        docs: Optional[DocList] = None,
        db_config=None,
        **kwargs,
    ):
        """Initialize InMemoryExactNNIndex"""
        super().__init__(db_config=db_config, **kwargs)
        self._runtime_config = self.RuntimeConfig()
        self._db_config = cast(InMemoryExactNNIndex.DBConfig, self._db_config)
        self._index_file_path = self._db_config.index_file_path

        if docs and self._index_file_path:
            raise ValueError(
                'Initialize `InMemoryExactNNIndex` with either `docs` or '
                '`index_file_path`, not both. Provide `docs` for a fresh index, or '
                '`index_file_path` to use an existing file.'
            )

        if self._index_file_path:
            if os.path.exists(self._index_file_path):
                self._logger.info(
                    f'Loading index from a binary file: {self._index_file_path}'
                )
                self._docs = DocList.__class_getitem__(
                    cast(Type[BaseDoc], self._schema)
                ).load_binary(file=self._index_file_path)

                data_by_columns = self._get_col_value_dict(self._docs)
                self._update_subindex_data(self._docs)
                self._index_subindex(data_by_columns)

            else:
                self._logger.warning(
                    f'Index file does not exist: {self._index_file_path}. '
                    f'Initializing empty InMemoryExactNNIndex.'
                )
                self._docs = DocList.__class_getitem__(
                    cast(Type[BaseDoc], self._schema)
                )()
        else:
            if docs:
                self._logger.info('Docs provided. Initializing with provided docs.')
                self._docs = docs
            else:
                self._logger.info(
                    'No docs or index file provided. Initializing empty InMemoryExactNNIndex.'
                )
                self._docs = DocList.__class_getitem__(
                    cast(Type[BaseDoc], self._schema)
                )()

        self._embedding_map: Dict[str, Tuple[AnyTensor, Optional[List[int]]]] = {}
        self._ids_to_positions: Dict[str, int] = {}

    def python_type_to_db_type(self, python_type: Type) -> Any:
        """Map python type to database type.
        Takes any python type and returns the corresponding database column type.

        :param python_type: a python type.
        :return: the corresponding database column type,
            or None if ``python_type`` is not supported.
        """
        pass

    @property
    def out_schema(self) -> Type[BaseDoc]:
        """Return the original schema (without the parent_id from new_schema type)"""
        pass

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(self, query: Optional[List[Tuple[str, Dict]]] = None):
            super().__init__()
            # list of tuples (method name, kwargs)
            self._queries: List[Tuple[str, Dict]] = query or []

        def build(self, *args, **kwargs) -> Any:
            """Build the query object."""
            pass

        find = _collect_query_args('find')
        find_batched = _collect_query_args('find_batched')
        filter = _collect_query_args('filter')
        filter_batched = _raise_not_supported('find_batched')
        text_search = _raise_not_supported('text_search')
        text_search_batched = _raise_not_supported('text_search')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Dataclass that contains all "static" configurations of InMemoryExactNNIndex."""

        index_file_path: Optional[str] = None
        default_column_config: Dict[Type, Dict[str, Any]] = field(
            default_factory=lambda: defaultdict(
                dict,
                {
                    AbstractTensor: {'space': 'cosine_sim'},
                },
            )
        )

    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        """Dataclass that contains all "dynamic" configurations of InMemoryExactNNIndex."""

        pass

    def index(self, docs: Union[BaseDoc, Sequence[BaseDoc]], **kwargs):
        """index Documents into the index.

        !!! note
            Passing a sequence of Documents that is not a DocList
            (such as a List of Docs) comes at a performance penalty.
            This is because the Index needs to check compatibility between itself and
            the data. With a DocList as input this is a single check; for other inputs
            compatibility needs to be checked for every Document individually.

        :param docs: Documents to index.
        """
        # implementing the public option because conversion to column dict is not needed
        docs = self._validate_docs(docs)
        ids_to_positions = self._get_ids_to_positions()
        for doc in docs:
            if doc.id in ids_to_positions:
                self._docs[ids_to_positions[doc.id]] = doc
            else:
                self._docs.append(doc)
                self._ids_to_positions[str(doc.id)] = len(self._ids_to_positions)

        # Add parent_id to all sub-index documents and store sub-index documents
        data_by_columns = self._get_col_value_dict(docs)
        self._update_subindex_data(docs)
        self._index_subindex(data_by_columns)

        self._rebuild_embedding()

    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        raise NotImplementedError

    def num_docs(self) -> int:
        """
        Get the number of documents.
        """
        return len(self._docs)

    def _rebuild_embedding(self):
        """
        Reconstructs the embeddings map for each field. This is performed to store pre-stacked
        embeddings, thereby optimizing performance by avoiding repeated stacking of embeddings.

        Note: '_embedding_map' is a dictionary mapping fields to their corresponding embeddings.
        """
        if self._is_index_empty:
            self._embedding_map = dict()
        else:
            for field_, embedding in self._embedding_map.items():
                self._embedding_map[field_] = _extract_embeddings(self._docs, field_)

    def _del_items(self, doc_ids: Sequence[str]):
        """Delete Documents from the index.

        :param doc_ids: ids to delete from the Document Store
        """
        pass

    def _ori_items(self, doc: BaseDoc) -> BaseDoc:
        """
        The Indexer's backend stores parent_id to support nested data. However,
        this method enables us to retrieve the original items in their original
        type, which is what the user interacts with.

        :param doc: The input document in New_Schema format from the Indexer's backend.
        :return: The input document with its original schema.
        """

        ori_doc = _shallow_copy_doc(doc)
        for field_name, type_, _ in self._flatten_schema(
            cast(Type[BaseDoc], self.out_schema)
        ):
            if safe_issubclass(type_, AnyDocArray):
                _list = getattr(ori_doc, field_name)
                for i, nested_doc in enumerate(_list):
                    sub_indexer: InMemoryExactNNIndex = cast(
                        InMemoryExactNNIndex, self._subindices[field_name]
                    )
                    nested_doc = self._subindices[field_name]._ori_schema(
                        **nested_doc.__dict__
                    )

                    _list[i] = sub_indexer._ori_items(nested_doc)

        return ori_doc

    def _get_items(
        self, doc_ids: Sequence[str], raw: bool = False
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        """Get Documents from the index, by `id`.
        If no document is found, a KeyError is raised.

        :param doc_ids: ids to get from the Document index
        :param raw: if raw, output the new_schema type (with parent id)
        :return: Sequence of Documents, sorted corresponding to the order of `doc_ids`.
            Duplicate `doc_ids` can be omitted in the output.
        """

        out_docs = []
        ids_to_positions = self._get_ids_to_positions()
        for doc_id in doc_ids:
            if doc_id not in ids_to_positions:
                continue
            doc = self._docs[ids_to_positions[doc_id]]
            if raw:
                out_docs.append(doc)
            else:
                ori_doc = self._ori_items(doc)
                schema_cls = cast(Type[BaseDoc], self.out_schema)
                new_doc = schema_cls(**ori_doc.__dict__)
                out_docs.append(new_doc)

        return out_docs

    def execute_query(self, query: List[Tuple[str, Dict]], *args, **kwargs) -> Any:
        """
        Execute a query on the InMemoryExactNNIndex.

        Can take two kinds of inputs:

        1. A native query of the underlying database. This is meant as a passthrough so that you
        can enjoy any functionality that is not available through the Document index API.
        2. The output of this Document index' `QueryBuilder.build()` method.

        :param query: the query to execute
        :param args: positional arguments to pass to the query
        :param kwargs: keyword arguments to pass to the query
        :return: the result of the query
        """
        pass

    def _find_and_filter(self, query: List[Tuple[str, Dict]]) -> FindResult:
        """
        The function executes search operations such as 'find' and 'filter' in the order
        they appear in the query. The 'find' operation performs a vector similarity search.
        The 'filter' operation filters out documents based on a filter query.
        The documents are finally sorted based on their scores.

        :param query: The query to execute.
        :return: A tuple of retrieved documents and their scores.
        """
        pass

    def find(
        self,
        query: Union[AnyTensor, BaseDoc],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResult:
        """Find Documents in the index using nearest-neighbor search.

        :param query: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.)
            with a single axis, or a Document
        :param search_field: name of the field to search on.
            Documents in the index are retrieved based on this similarity
            of this field to the query.
        :param limit: maximum number of Documents to return
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug(f'Executing `find` for search field {search_field}')
        self._validate_search_field(search_field)

        if self._is_index_empty:
            return FindResult(documents=[], scores=[])  # type: ignore

        config = self._column_infos[search_field].config

        docs, scores = find(
            index=self._docs,
            query=query,
            search_field=search_field,
            limit=limit,
            metric=config['space'],
            cache=self._embedding_map,
        )

        docs_ = []
        for doc in docs:
            ori_doc = self._ori_items(doc)
            schema_cls = cast(Type[BaseDoc], self.out_schema)
            docs_.append(schema_cls(**ori_doc.__dict__))

        docs_with_schema = DocList.__class_getitem__(
            cast(Type[BaseDoc], self.out_schema)
        )(docs_)

        return FindResult(documents=docs_with_schema, scores=scores)

    def _find(
        self, query: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResult:
        raise NotImplementedError

    def find_batched(
        self,
        queries: Union[AnyTensor, DocList],
        search_field: str = '',
        limit: int = 10,
        **kwargs,
    ) -> FindResultBatched:
        """Find Documents in the index using nearest-neighbor search.

        :param queries: query vector for KNN/ANN search.
            Can be either a tensor-like (np.array, torch.Tensor, etc.) with a,
            or a DocList.
            If a tensor-like is passed, it should have shape (batch_size, vector_dim)
        :param search_field: name of the field to search on.
            Documents in the index are retrieved based on this similarity
            of this field to the query.
        :param limit: maximum number of documents to return per query
        :return: a named tuple containing `documents` and `scores`
        """
        self._logger.debug(f'Executing `find_batched` for search field {search_field}')
        self._validate_search_field(search_field)

        if self._is_index_empty:
            return FindResultBatched(documents=[], scores=[])  # type: ignore

        config = self._column_infos[search_field].config

        find_res = find_batched(
            index=self._docs,
            query=cast(NdArray, queries),
            search_field=search_field,
            limit=limit,
            metric=config['space'],
            cache=self._embedding_map,
        )

        return find_res

    def _find_batched(
        self, queries: np.ndarray, limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        raise NotImplementedError

    def filter(
        self,
        filter_query: Any,
        limit: int = 10,
        **kwargs,
    ) -> DocList:
        """Find documents in the index based on a filter query

        :param filter_query: the filter query to execute following the query
            language of
        :param limit: maximum number of documents to return
        :return: a DocList containing the documents that match the filter query
        """
        pass

    def _filter(self, filter_query: Any, limit: int) -> Union[DocList, List[Dict]]:
        raise NotImplementedError

    def _filter_batched(
        self, filter_queries: Any, limit: int
    ) -> Union[List[DocList], List[List[Dict]]]:
        raise NotImplementedError(f'{type(self)} does not support filtering.')

    def _text_search(
        self, query: str, limit: int, search_field: str = ''
    ) -> _FindResult:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def _text_search_batched(
        self, queries: Sequence[str], limit: int, search_field: str = ''
    ) -> _FindResultBatched:
        raise NotImplementedError(f'{type(self)} does not support text search.')


    def persist(self, file: Optional[str] = None) -> None:
        """Persist InMemoryExactNNIndex into a binary file."""
        pass

    def _get_root_doc_id(self, id: str, root: str, sub: str) -> str:
        """Get the root_id given the id of a subindex Document and the root and subindex name

        :param id: id of the subindex Document
        :param root: root index name
        :param sub: subindex name
        :return: the root_id of the Document
        """
        pass

    def _get_ids_to_positions(self) -> Dict[str, int]:
        """
        Obtains a mapping between document IDs and their respective positions
        within the DocList. If this mapping hasn't been initialized, it will be created.

        :return: A dictionary mapping each document ID to its corresponding position.
        """
        if not self._ids_to_positions:
            self._update_ids_to_positions()
        return self._ids_to_positions

    def _update_ids_to_positions(self) -> None:
        """
        Generates or updates the mapping between document IDs and their corresponding
        positions within the DocList.
        """
        self._ids_to_positions = {doc.id: pos for pos, doc in enumerate(self._docs)}
