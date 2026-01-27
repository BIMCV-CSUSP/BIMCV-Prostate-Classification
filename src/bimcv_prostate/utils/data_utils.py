from __future__ import annotations

import ast
from typing import Iterable, Sequence, Tuple

from pandas import DataFrame


def parse_input_shape(input_shape: object) -> Tuple[int, int, int]:
    if isinstance(input_shape, tuple):
        return input_shape
    if isinstance(input_shape, list):
        return tuple(input_shape)
    if isinstance(input_shape, str):
        return tuple(ast.literal_eval(input_shape))
    raise TypeError(f"Unsupported input_shape type: {type(input_shape)}")


def apply_path_rewrites(
    df: DataFrame,
    columns: Sequence[str],
    rewrites: Iterable[Tuple[str, str]] | None,
) -> DataFrame:
    if not rewrites:
        return df
    for column in columns:
        for src, dst in rewrites:
            df[column] = df[column].str.replace(src, dst, regex=False)
    return df


def ensure_partition(df: DataFrame, groupby_key: str, partition: str) -> None:
    if partition not in df[groupby_key].unique():
        raise KeyError(f"Partition '{partition}' not found in column '{groupby_key}'.")
