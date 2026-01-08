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
        rand_prob: float = 0.5,
        partition_column: str = "partition",
        config: dict = config_default,
    ):
        df = read_csv(path, sep=sep)

        df['filepath_t2w_cropped'] = df['filepath_t2w_cropped'].apply(lambda x: x.replace("/nvmescratch/ceib/Prostate", "/mnt/datalake/FISABIO_datalake"))
        df['filepath_adc_cropped'] = df['filepath_adc_cropped'].apply(lambda x: x.replace("/nvmescratch/ceib/Prostate", "/mnt/datalake/FISABIO_datalake"))
        df['filepath_hbv_cropped'] = df['filepath_hbv_cropped'].apply(lambda x: x.replace("/nvmescratch/ceib/Prostate", "/mnt/datalake/FISABIO_datalake"))
        
        n_classes = len(unique(df["label"].values))

        self.groupby = df.groupby(partition_column)
        print(f"Number of partitions: {len(self.groupby)}")
        print(f"Partitions: {self.groupby.groups.keys()}")

        self._class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique(self.groupby.get_group("train")["label"].values),
            y=self.groupby.get_group("train")["label"].values,
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
            transforms.ScaleIntensityRangePercentilesd(keys=['adc'], lower=5, upper=99.5,b_min=0.0, b_max=1.0,clip=True),
            transforms.ScaleIntensityd(keys=['t2','dwi_0'], minv=0.0, maxv=1.0, allow_missing_keys=True),
            transforms.ConcatItemsd(keys=['t2','dwi_0','adc'], name="image", dim=0),
            transforms.SelectItemsd(keys=["image", "label"]),
            transforms.RandFlipd(keys=["image"], spatial_axis=(0), prob=0.2),
            transforms.RandRotate90d(keys=["image"], prob=0.2),
            transforms.RandAffined(keys=["image"], rotate_range=[0.1, 0.1, 0.1], translate_range=[0.1, 0.1, 0.1], prob=0.2),
            transforms.RandGaussianNoised(keys=["image"], std=0.1, mean=0.0, prob=0.2)
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
            transforms.ScaleIntensityRangePercentilesd(keys=['adc'], lower=5, upper=99.5,b_min=0.0, b_max=1.0,clip=True),
            transforms.ScaleIntensityd(keys=['t2','dwi_0'], minv=0.0, maxv=1.0, allow_missing_keys=True),
            transforms.ConcatItemsd(keys=['t2','dwi_0','adc'], name="image", dim=0),
            transforms.SelectItemsd(keys=["image", "label"])
        ])
        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        if partition == "None":
            print("Partition not found")
            return None
        data = [
            {"t2": t2, "adc": adc, "dwi": dwi, "label": label}
            for t2, adc, dwi, label in zip(
                self.groupby.get_group(partition)["filepath_t2w_cropped"].values,
                self.groupby.get_group(partition)["filepath_adc_cropped"].values,
                self.groupby.get_group(partition)["filepath_hbv_cropped"].values,
                one_hot(as_tensor(self.groupby.get_group(partition)["label"].values, dtype=int)).float(),
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