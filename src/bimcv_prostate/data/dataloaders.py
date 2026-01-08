from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from monai.data import CacheDataset, DataLoader
from numpy import array, float32, unique
from pandas import DataFrame, read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

from bimcv_prostate.data.transforms import build_image_transforms
from bimcv_prostate.utils.data_utils import apply_path_rewrites, ensure_partition, parse_input_shape


@dataclass(frozen=True)
class ClinicalFilter:
    columns: Sequence[str]
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99


class ProstateImageDataLoader:
    def __init__(
        self,
        path: str,
        sep: str = ",",
        img_columns: Sequence[str] = ("t2", "adc", "dwi"),
        csv_image_columns: Optional[Sequence[str]] = None,
        label_column: str = "csPC",
        partition_column: str = "partition",
        input_shape: object = (128, 128, 32),
        test_run: bool = False,
        rand_prob: float = 0.5,
        config: Optional[Dict] = None,
        adc_percentile_scale: bool = True,
        path_rewrites: Optional[Iterable[Tuple[str, str]]] = None,
        clinical_filter: Optional[ClinicalFilter] = None,
        skip_partitions: Optional[Sequence[str]] = None,
    ):
        del rand_prob
        df = read_csv(path, sep=sep)

        csv_columns = list(csv_image_columns) if csv_image_columns is not None else list(img_columns)
        apply_path_rewrites(df, csv_columns, path_rewrites)
        if clinical_filter is not None:
            df = self._apply_clinical_filter(df, clinical_filter)

        self._label_column = label_column
        self._partition_column = partition_column
        self._img_columns = list(img_columns)
        self._csv_image_columns = csv_columns
        self._input_shape = parse_input_shape(input_shape)
        self._test_run = test_run
        self._config_args = dict(config) if config else {}
        self._num_classes = len(unique(df[label_column].values))
        self._skip_partitions = set(skip_partitions or [])

        self._groupby = df.groupby(partition_column)
        ensure_partition(df, partition_column, "train")

        self._class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique(self._groupby.get_group("train")[label_column].values),
            y=self._groupby.get_group("train")[label_column].values,
        )

        self._train_transforms = build_image_transforms(
            img_columns=self._img_columns,
            input_shape=self._input_shape,
            augment=True,
            adc_percentile_scale=adc_percentile_scale,
            include_numeric=False,
        )
        self._val_transforms = build_image_transforms(
            img_columns=self._img_columns,
            input_shape=self._input_shape,
            augment=False,
            adc_percentile_scale=adc_percentile_scale,
            include_numeric=False,
        )

    def _apply_clinical_filter(self, df: DataFrame, config: ClinicalFilter) -> DataFrame:
        filtered = df[list(config.columns)].dropna()
        lower_bound = filtered.quantile(config.lower_quantile)
        upper_bound = filtered.quantile(config.upper_quantile)
        cleaned = filtered[(filtered >= lower_bound) & (filtered <= upper_bound)].dropna()
        return df.iloc[cleaned.index]

    def __call__(self, partition: str):
        if partition in self._skip_partitions:
            return None
        ensure_partition(self._groupby.obj, self._partition_column, partition)
        data = [
            {"t2": t2, "adc": adc, "dwi": dwi, "label": label}
            for t2, adc, dwi, label in zip(
                self._groupby.get_group(partition)[self._csv_image_columns[0]].values,
                self._groupby.get_group(partition)[self._csv_image_columns[1]].values,
                self._groupby.get_group(partition)[self._csv_image_columns[2]].values,
                one_hot(
                    as_tensor(self._groupby.get_group(partition)[self._label_column].values, dtype=int),
                    num_classes=self._num_classes,
                ).float(),
            )
        ]
        if self._test_run:
            data = data[:16]

        transforms = self._train_transforms if partition == "train" else self._val_transforms
        dataset = CacheDataset(data=data, transform=transforms, num_workers=7)

        loader_args = dict(self._config_args)
        if partition != "train":
            loader_args["shuffle"] = False
        return DataLoader(dataset, **loader_args)

    @property
    def class_weights(self):
        return self._class_weights


class ProstateMultimodalDataLoader(ProstateImageDataLoader):
    def __init__(
        self,
        path: str,
        sep: str = ",",
        img_columns: Sequence[str] = ("t2", "adc", "dwi"),
        csv_image_columns: Optional[Sequence[str]] = None,
        label_column: str = "csPC",
        partition_column: str = "partition",
        input_shape: object = (128, 128, 32),
        test_run: bool = False,
        rand_prob: float = 0.5,
        config: Optional[Dict] = None,
        adc_percentile_scale: bool = False,
        path_rewrites: Optional[Iterable[Tuple[str, str]]] = None,
        clinical_filter: Optional[ClinicalFilter] = None,
        clinical_columns: Sequence[str] = ("ED", "PSA", "VP", "PSA_density"),
        skip_partitions: Optional[Sequence[str]] = None,
    ):
        self._clinical_columns = list(clinical_columns)
        super().__init__(
            path=path,
            sep=sep,
            img_columns=img_columns,
            csv_image_columns=csv_image_columns,
            label_column=label_column,
            partition_column=partition_column,
            input_shape=input_shape,
            test_run=test_run,
            rand_prob=rand_prob,
            config=config,
            adc_percentile_scale=adc_percentile_scale,
            path_rewrites=path_rewrites,
            clinical_filter=clinical_filter,
            skip_partitions=skip_partitions,
        )

        self._train_transforms = build_image_transforms(
            img_columns=self._img_columns,
            input_shape=self._input_shape,
            augment=True,
            adc_percentile_scale=adc_percentile_scale,
            include_numeric=True,
        )
        self._val_transforms = build_image_transforms(
            img_columns=self._img_columns,
            input_shape=self._input_shape,
            augment=False,
            adc_percentile_scale=adc_percentile_scale,
            include_numeric=True,
        )

    def __call__(self, partition: str):
        if partition in self._skip_partitions:
            return None
        ensure_partition(self._groupby.obj, self._partition_column, partition)
        clinical_values = array(
            self._groupby.get_group(partition)[self._clinical_columns].values,
            dtype=float32,
        )
        data = [
            {"t2": t2, "adc": adc, "dwi": dwi, "label": label, "numeric": clinical}
            for t2, adc, dwi, label, clinical in zip(
                self._groupby.get_group(partition)[self._csv_image_columns[0]].values,
                self._groupby.get_group(partition)[self._csv_image_columns[1]].values,
                self._groupby.get_group(partition)[self._csv_image_columns[2]].values,
                one_hot(
                    as_tensor(self._groupby.get_group(partition)[self._label_column].values, dtype=int),
                    num_classes=self._num_classes,
                ).float(),
                clinical_values,
            )
        ]
        if self._test_run:
            data = data[:16]

        transforms = self._train_transforms if partition == "train" else self._val_transforms
        dataset = CacheDataset(data=data, transform=transforms, num_workers=7)

        loader_args = dict(self._config_args)
        if partition != "train":
            loader_args["shuffle"] = False
        return DataLoader(dataset, **loader_args)
