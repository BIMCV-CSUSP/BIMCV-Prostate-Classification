from __future__ import annotations

from typing import Dict, Optional, Sequence

from bimcv_prostate.data.dataloaders import (
    ClinicalFilter,
    ProstateImageDataLoader,
    ProstateMultimodalDataLoader as BaseProstateMultimodalDataLoader,
)


class DataloaderImages:
    def __init__(
        self,
        path: str,
        sep: str = ",",
        classes: Sequence[str] = ("noCsPCa", "CsPCa"),
        img_columns: Sequence[str] = ("t2", "adc", "dwi"),
        test_run: bool = False,
        input_shape: object = (128, 128, 32),
        rand_prob: float = 0.5,
        partition_column: str = "partition",
        config: Optional[Dict] = None,
    ):
        del classes
        clinical_filter = ClinicalFilter(columns=["ED", "PSA", "VP", "PSA_density", "csPC"])
        self._loader = ProstateImageDataLoader(
            path=path,
            sep=sep,
            img_columns=img_columns,
            csv_image_columns=("image_t2", "image_adc", "image_dwi"),
            label_column="csPC",
            partition_column=partition_column,
            input_shape=input_shape,
            test_run=test_run,
            rand_prob=rand_prob,
            config=config,
            adc_percentile_scale=False,
            clinical_filter=clinical_filter,
        )

    def __call__(self, partition: str):
        if partition == "test":
            return None
        return self._loader(partition)

    @property
    def class_weights(self):
        return self._loader.class_weights


class ProstateMultimodalDataLoader:
    def __init__(
        self,
        path: str,
        sep: str = ",",
        classes: Sequence[str] = ("noCsPCa", "CsPCa"),
        img_columns: Sequence[str] = ("t2", "adc", "dwi"),
        test_run: bool = False,
        input_shape: object = (128, 128, 32),
        rand_prob: float = 0.5,
        partition_column: str = "partition",
        config: Optional[Dict] = None,
    ):
        del classes
        clinical_filter = ClinicalFilter(columns=["ED", "PSA", "VP", "PSA_density", "csPC"])
        self._loader = BaseProstateMultimodalDataLoader(
            path=path,
            sep=sep,
            img_columns=img_columns,
            csv_image_columns=("image_t2", "image_adc", "image_dwi"),
            label_column="csPC",
            partition_column=partition_column,
            input_shape=input_shape,
            test_run=test_run,
            rand_prob=rand_prob,
            config=config,
            adc_percentile_scale=False,
            clinical_filter=clinical_filter,
            clinical_columns=("ED", "PSA", "VP", "PSA_density"),
        )

    def __call__(self, partition: str):
        if partition == "test":
            return None
        return self._loader(partition)

    @property
    def class_weights(self):
        return self._loader.class_weights
