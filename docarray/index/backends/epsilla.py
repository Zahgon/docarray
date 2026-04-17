import copy
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import (
    Any,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from pyepsilla import cloud, vectordb

from docarray import BaseDoc, DocList
from docarray.index.abstract import (
    BaseDocIndex,
    _FindResultBatched,
    _raise_not_composable,
    _raise_not_supported,
)
from docarray.typing import ID, NdArray
from docarray.typing.tensor.abstract_tensor import AbstractTensor
from docarray.utils._internal._typing import safe_issubclass
from docarray.utils.find import _FindResult

TSchema = TypeVar('TSchema', bound=BaseDoc)


class EpsillaDocumentIndex(BaseDocIndex, Generic[TSchema]):
    def __init__(self, db_config=None, **kwargs):
        # will set _db_config from args / kwargs
        super().__init__(db_config=db_config, **kwargs)

        self._db_config: EpsillaDocumentIndex.DBConfig = cast(
            EpsillaDocumentIndex.DBConfig, self._db_config
        )
        self._db_config.validate_config()
        self._validate_column_info()

        self._table_name = (
            self._db_config.table_name
            if self._db_config.table_name
            else self._schema.__name__
        )

        if self._db_config.is_self_hosted:
            self._db = vectordb.Client(
                protocol=self._db_config.protocol,
                host=self._db_config.host,
                port=self._db_config.port,
            )
            status_code, response = self._db.load_db(
                db_name=self._db_config.db_name,
                db_path=self._db_config.db_path,
            )

            if status_code != HTTPStatus.OK:
                if status_code == HTTPStatus.CONFLICT:
                    self._logger.info(f'{self._db_config.db_name} already loaded.')
                else:
                    raise IOError(
                        f"Failed to load database {self._db_config.db_name}. "
                        f"Error code: {status_code}. Error message: {response}."
                    )
            self._db.use_db(self._db_config.db_name)

            status_code, response = self._db.list_tables()
            if status_code != HTTPStatus.OK:
                raise IOError(
                    f"Failed to list tables. "
                    f"Error code: {status_code}. Error message: {response}."
                )

            if self._table_name not in response["result"]:
                self._create_table_self_hosted()
        else:
            self._client = cloud.Client(
                project_id=self._db_config.cloud_project_id,
                api_key=self._db_config.api_key,
            )
            self._db = self._client.vectordb(self._db_config.cloud_db_id)

            status_code, response = self._db.list_tables()
            if status_code != HTTPStatus.OK:
                raise IOError(
                    f"Failed to list tables. "
                    f"Error code: {status_code}. Error message: {response}."
                )

            # Epsilla cloud requires table to be created in the web UI before inserting data
            # It does not support creating tables from Python client yet.


    def _create_table_self_hosted(self):
        """Use _column_infos to create a table in the database."""
        pass

    @dataclass
    class Query:
        """Dataclass describing a query."""

        vector_field: Optional[str]
        vector_query: Optional[NdArray]
        filter: Optional[str]
        limit: int

    class QueryBuilder(BaseDocIndex.QueryBuilder):
        def __init__(
            self,
            vector_search_field: Optional[str] = None,
            vector_queries: Optional[List[NdArray]] = None,
            filter: Optional[str] = None,
        ):
            self._vector_search_field: Optional[str] = vector_search_field
            self._vector_queries: List[NdArray] = vector_queries or []
            self._filter: Optional[str] = filter

        def find(self, query: NdArray, search_field: str = ''):
            if self._vector_search_field and self._vector_search_field != search_field:
                raise ValueError(
                    f'Trying to call .find for search_field = {search_field}, but '
                    f'previously {self._vector_search_field} was used. Only a single '
                    f'field might be used in chained calls.'
                )
            return EpsillaDocumentIndex.QueryBuilder(
                vector_search_field=search_field,
                vector_queries=self._vector_queries + [query],
                filter=self._filter,
            )



        find_batched = _raise_not_composable('find_batched')
        filter_batched = _raise_not_composable('filter_batched')
        text_search = _raise_not_supported('text_search')
        text_search_batched = _raise_not_supported('text_search_batched')

    @dataclass
    class DBConfig(BaseDocIndex.DBConfig):
        """Static configuration for EpsillaDocumentIndex"""

        # default value is the schema type name
        table_name: Optional[str] = None

        # Indicator for self-hosted or cloud version
        is_self_hosted: bool = False

        # self-hosted version uses the following configs
        protocol: Optional[str] = None
        host: Optional[str] = None
        port: Optional[int] = 8888
        db_path: Optional[str] = None
        db_name: Optional[str] = None

        # cloud version uses the following configs
        cloud_project_id: Optional[str] = None
        cloud_db_id: Optional[str] = None
        api_key: Optional[str] = None

        default_column_config: Dict[Any, Dict[str, Any]] = field(
            default_factory=lambda: {
                'TINYINT': {},
                'SMALLINT': {},
                'INT': {},
                'BIGINT': {},
                'FLOAT': {},
                'DOUBLE': {},
                'STRING': {},
                'BOOL': {},
                'JSON': {},
                'VECTOR_FLOAT': {},
            }
        )




    @dataclass
    class RuntimeConfig(BaseDocIndex.RuntimeConfig):
        # No dynamic config used
        pass




    def _index(self, column_to_data: Dict[str, Generator[Any, None, None]]):
        self._index_subindex(column_to_data)

        rows = list(self._transpose_col_value_dict(column_to_data))
        normalized_rows = []
        for row in rows:
            normalized_row = {}
            for key, value in row.items():
                if isinstance(value, NdArray):
                    normalized_row[key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    normalized_row[key] = value.tolist()
                else:
                    normalized_row[key] = value
            normalized_rows.append(normalized_row)

        status_code, response = self._db.insert(
            table_name=self._table_name, records=normalized_rows
        )

        if status_code != HTTPStatus.OK:
            raise IOError(
                f"Failed to insert documents. "
                f"Error code: {status_code}. Error message: {response}."
            )

    def num_docs(self) -> int:
        raise NotImplementedError

    @property
    def _is_index_empty(self) -> bool:
        """
        Check if index is empty by comparing the number of documents to zero.
        :return: True if the index is empty, False otherwise.
        """
        pass


    def _get_items(
        self, doc_ids: Sequence[str]
    ) -> Union[Sequence[TSchema], Sequence[Dict[str, Any]]]:
        status_code, response = self._db.get(
            table_name=self._table_name,
            primary_keys=list(doc_ids),
        )
        if status_code != HTTPStatus.OK:
            raise IOError(
                f"Failed to get documents with ids {doc_ids}. "
                f"Error code: {status_code}. Error message: {response}."
            )
        return response['result']



    def _find(
        self,
        query: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        query_batched = np.expand_dims(query, axis=0)
        docs, scores = self._find_batched(
            queries=query_batched, limit=limit, search_field=search_field
        )
        return _FindResult(documents=docs[0], scores=scores[0])

    def _find_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        return self._find_with_filter_batched(
            queries=queries, limit=limit, search_field=search_field
        )

    def _find_with_filter_batched(
        self,
        queries: np.ndarray,
        limit: int,
        search_field: str,
        filter: Optional[str] = None,
    ) -> _FindResultBatched:
        if search_field == '':
            raise ValueError(
                'EpsillaDocumentIndex requires a search_field to be specified.'
            )

        responses = []
        for query in queries:
            status_code, response = self._db.query(
                table_name=self._table_name,
                query_field=search_field,
                limit=limit,
                filter=filter if filter is not None else '',
                query_vector=query.tolist(),
                with_distance=True,
            )

            if status_code != HTTPStatus.OK:
                raise IOError(
                    f"Failed to find documents with query {query}. "
                    f"Error code: {status_code}. Error message: {response}."
                )

            results = response['result']
            scores = NdArray._docarray_from_native(
                np.array([result['@distance'] for result in results])
            )
            documents = []
            for result in results:
                doc = copy.copy(result)
                del doc["@distance"]
                documents.append(doc)

            responses.append((documents, scores))

        return _FindResultBatched(
            documents=[r[0] for r in responses],
            scores=[r[1] for r in responses],
        )

    def _filter(
        self,
        filter_query: str,
        limit: int,
    ) -> Union[DocList, List[Dict]]:
        query_batched = [filter_query]
        docs = self._filter_batched(filter_queries=query_batched, limit=limit)
        return docs[0]

    def _filter_batched(
        self,
        filter_queries: str,
        limit: int,
    ) -> Union[List[DocList], List[List[Dict]]]:
        responses = []
        for filter_query in filter_queries:
            status_code, response = self._db.get(
                table_name=self._table_name,
                limit=limit,
                filter=filter_query,
            )

            if status_code != HTTPStatus.OK:
                raise IOError(
                    f"Failed to find documents with filter {filter_query}. "
                    f"Error code: {status_code}. Error message: {response}."
                )

            results = response['result']
            responses.append(results)

        return responses

    def _text_search(
        self,
        query: str,
        limit: int,
        search_field: str = '',
    ) -> _FindResult:
        raise NotImplementedError(f'{type(self)} does not support text search.')

    def _text_search_batched(
        self,
        queries: Sequence[str],
        limit: int,
        search_field: str = '',
    ) -> _FindResultBatched:
        raise NotImplementedError(f'{type(self)} does not support text search.')
