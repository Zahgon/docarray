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
from typing import Any, Dict, List, Optional, Union

from docarray.utils._internal.query_language.lookup import (
    LookupLeaf,
    LookupNode,
    LookupTreeElem,
    Q,
)

LOGICAL_OPERATORS: Dict[str, Union[str, bool]] = {
    '$and': 'and',
    '$or': 'or',
    '$not': True,
}

COMPARISON_OPERATORS = {
    '$lt': 'lt',
    '$gt': 'gt',
    '$lte': 'lte',
    '$gte': 'gte',
    '$eq': 'exact',
    '$neq': 'neq',
    '$exists': 'exists',
}

REGEX_OPERATORS = {'$regex': 'regex'}

ARRAY_OPERATORS = {'$size': 'size'}

MEMBERSHIP_OPERATORS = {'$in': 'in', '$nin': 'nin'}

SUPPORTED_OPERATORS = {
    **COMPARISON_OPERATORS,
    **ARRAY_OPERATORS,
    **REGEX_OPERATORS,
    **MEMBERSHIP_OPERATORS,
}




class QueryParser:
    """A class to parse dict condition to lookup query."""

    def __init__(self, conditions: Union[Dict, List] = {}):
        self.conditions = conditions
        self.lookup_groups = _parse_lookups(self.conditions)

    def evaluate(self, doc: Any) -> bool:
        return self.lookup_groups.evaluate(doc) if self.lookup_groups else True

    def __call__(self, doc: Any) -> bool:
        return self.evaluate(doc)
