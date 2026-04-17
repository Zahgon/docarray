from typing import Any, Callable, Dict, Type

import orjson

from docarray.utils._internal.pydantic import is_pydantic_v2

if not is_pydantic_v2:
    from pydantic.json import ENCODERS_BY_TYPE
else:
    ENCODERS_BY_TYPE: Dict[Type[Any], Callable[[Any], Any]] = {
        bytes: lambda o: o.decode(),
        frozenset: list,
        set: list,
    }


def _default_orjson(obj):
    """
    default option for orjson dumps.
    :param obj:
    :return: return a json compatible object
    """
    pass


def orjson_dumps(v, *, default=None) -> bytes:
    # dumps to bytes using orjson
    return orjson.dumps(v, default=_default_orjson, option=orjson.OPT_SERIALIZE_NUMPY)


def orjson_dumps_and_decode(v, *, default=None) -> str:
    # dumps to str using orjson
    return orjson_dumps(v, default=default).decode()
