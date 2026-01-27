from __future__ import annotations

from typing import Sequence, Tuple

from monai import transforms


def build_image_transforms(
    img_columns: Sequence[str],
    input_shape: Tuple[int, int, int],
    augment: bool,
    adc_percentile_scale: bool,
    include_numeric: bool,
) -> transforms.Compose:
    ops = [
        transforms.LoadImaged(keys=img_columns, image_only=True, ensure_channel_first=True),
        transforms.ResampleToMatchd(
            keys=["adc", "dwi"],
            key_dst="t2",
            mode=("bilinear", "bilinear"),
        ),
        transforms.SplitDimd(keys=["dwi"], keepdim=True),
        transforms.Resized(
            keys=["t2", "dwi_0", "adc"],
            spatial_size=input_shape,
            mode=("trilinear", "trilinear", "trilinear"),
        ),
    ]

    if adc_percentile_scale:
        ops.append(
            transforms.ScaleIntensityRangePercentilesd(
                keys=["adc"],
                lower=5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
        ops.append(
            transforms.ScaleIntensityd(
                keys=["t2", "dwi_0"],
                minv=0.0,
                maxv=1.0,
                allow_missing_keys=True,
            )
        )
    else:
        ops.append(
            transforms.ScaleIntensityd(
                keys=["t2", "dwi_0", "adc"],
                minv=0.0,
                maxv=1.0,
                allow_missing_keys=True,
            )
        )

    ops.append(transforms.ConcatItemsd(keys=["t2", "dwi_0", "adc"], name="image", dim=0))

    select_keys = ["image", "label"]
    if include_numeric:
        select_keys.append("numeric")
    ops.append(transforms.SelectItemsd(keys=select_keys))

    if include_numeric:
        ops.append(transforms.ToTensord(keys=["numeric"]))

    if augment:
        ops.extend(
            [
                transforms.RandFlipd(keys=["image"], spatial_axis=(0), prob=0.2),
                transforms.RandRotate90d(keys=["image"], prob=0.2),
                transforms.RandAffined(
                    keys=["image"],
                    rotate_range=[0.1, 0.1, 0.1],
                    translate_range=[0.1, 0.1, 0.1],
                    prob=0.2,
                ),
                transforms.RandGaussianNoised(keys=["image"], std=0.1, mean=0.0, prob=0.2),
            ]
        )

    return transforms.Compose(ops)
