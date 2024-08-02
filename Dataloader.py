import nibabel as nib
from bimcv_aikit.monai.transforms import DeleteBlackSlices
from monai import transforms
from monai.data import CacheDataset, DataLoader
from numpy import array, float32, unique
from pandas import read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

config_default = {}


class Dataloader:
    def __init__(
        self,
        path: str,
        sep: str = ",",
        classes: list = ["noCsPCa", "CsPCa"],
        img_columns=["t2", "adc", "dwi"],
        test_run: bool = False,
        input_shape: str = "(128, 128, 32)",
        rand_prob: int = 0.5,
        partition_column: str = "partition",
        config: dict = config_default,
    ):
        df = read_csv(path, sep=sep)

        n_classes = len(unique(df["csPC"].values))

        self.groupby = df.groupby(partition_column)

        self._class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique(self.groupby.get_group("train")["csPC"].values),
            y=self.groupby.get_group("train")["csPC"].values,
        )
        self.train_transforms = transforms.Compose([
            transforms.LoadImaged(keys=img_columns, image_only=True,ensure_channel_first=True),
            transforms.ResampleToMatchd(
                keys=["adc", "dwi"],
                key_dst="t2",
                mode=("bilinear", "bilinear"),
            ), # Resample images to t2 dimension
            transforms.SplitDimd(
                keys=["dwi"],
                keepdim=True,
            ), 
            transforms.Resized(
                keys=['t2','dwi_0','adc'],
                spatial_size=eval(input_shape),
                mode=("trilinear", "trilinear", "trilinear"),
            ),
            transforms.ScaleIntensityd(keys=['t2','dwi_0','adc'], minv=0.0, maxv=1.0, allow_missing_keys=True),
            transforms.ConcatItemsd(keys=['t2','dwi_0','adc'], name="image", dim=0),
            transforms.SelectItemsd(keys=["image", "label"])
        ])
        self.val_transforms = transforms.Compose([
            transforms.LoadImaged(keys=img_columns, image_only=True,ensure_channel_first=True),
            transforms.ResampleToMatchd(
                keys=["adc", "dwi"],
                key_dst="t2",
                mode=("bilinear", "bilinear"),
            ), # Resample images to t2 dimension
            transforms.SplitDimd(
                keys=["dwi"],
                keepdim=True,
            ), 
            transforms.Resized(
                keys=['t2','dwi_0','adc'],
                spatial_size=eval(input_shape),
                mode=("trilinear", "trilinear", "trilinear"),
            ),
            transforms.ScaleIntensityd(keys=['t2','dwi_0','adc'], minv=0.0, maxv=1.0, allow_missing_keys=True),
            transforms.ConcatItemsd(keys=['t2','dwi_0','adc'], name="image", dim=0),
            transforms.SelectItemsd(keys=["image", "label"])
        ])
        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        if partition == "test":
            return None
        data = [
            {"t2": t2, "adc": adc, "dwi": dwi, "label": label}
            for t2, adc, dwi, label in zip(
                self.groupby.get_group(partition)["image_t2"].values,
                self.groupby.get_group(partition)["image_adc"].values,
                self.groupby.get_group(partition)["image_dwi"].values,
                one_hot(as_tensor(self.groupby.get_group(partition)["csPC"].values, dtype=int)).float(),
            )
        ]
        if self.test_run:
            data = data[:16]
        if partition == "train":
            dataset = CacheDataset(data=data, transform=self.train_transforms, num_workers=7)
        else:
            dataset = CacheDataset(data=data, transform=self.val_transforms, num_workers=7)
            self.config_args["shuffle"] = False
        return DataLoader(dataset, **self.config_args)

    @property
    def class_weights(self):
        return self._class_weights