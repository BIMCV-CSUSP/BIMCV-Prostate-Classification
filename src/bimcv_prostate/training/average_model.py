from __future__ import annotations

import torch
from pathlib import Path
from typing import Dict, List
from monai.networks.nets import EfficientNetBN


def load_fold_checkpoints(paths: List[Path]) -> List[Dict]:
    ckpts = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu")  # keep on CPU for averaging
        state = ckpt.get("state_dict", ckpt)
        ckpts.append(state)
    return ckpts

def average_state_dicts(state_dicts: List[Dict[str, torch.Tensor]],
                        weights: List[float],
                        exclude_keys_substrings=("classifier", "fc", "last_linear")) -> Dict[str, torch.Tensor]:
    if len(state_dicts) == 0:
        raise ValueError("No state dicts provided.")
    if len(state_dicts) != len(weights):
        raise ValueError("Weights length must match number of state dicts.")
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()

    ref_keys = state_dicts[0].keys()
    avg_state = {}

    for k in ref_keys:
        if any(sub in k for sub in exclude_keys_substrings):
            continue

        tensors = []
        for sd in state_dicts:
            if k not in sd:
                raise ValueError(f"Key {k} missing in one state_dict.")
            t = sd[k]
            tensors.append(t)

        # If scalar long buffer (e.g., num_batches_tracked) just take first
        if tensors[0].dtype in (torch.int64, torch.int32, torch.uint8, torch.bool):
            avg_state[k] = tensors[0]
            continue

        stacked = torch.stack([t.to(torch.float32) for t in tensors], dim=0)  # (n, ...)
        # Broadcast weights
        avg = (w.view(-1, *([1] * (stacked.ndim - 1))) * stacked).sum(dim=0)
        avg_state[k] = avg.to(tensors[0].dtype)

    return avg_state

def build_new_model_from_folds(
    fold_ckpt_paths: List[str],
    fold_validation_scores: List[float],
    new_num_classes: int,
    model_name: str = "efficientnet-b0",
    in_channels: int = 1,
) -> EfficientNetBN:
    states = load_fold_checkpoints([Path(p) for p in fold_ckpt_paths])
    avg_backbone = average_state_dicts(states, fold_validation_scores)

    model = EfficientNetBN(
        model_name=model_name,
        spatial_dims=3,
        in_channels=in_channels,
        num_classes=new_num_classes,
    )

    model_state = model.state_dict()
    loaded = 0
    skipped = []
    for k, v in avg_backbone.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(k)
    model.load_state_dict(model_state, strict=False)
    print(f"Loaded averaged params: {loaded}. Skipped (head/mismatch): {len(skipped)}")

    # Re-init classifier head
    head_param_names = []
    for name, param in model.named_parameters():
        if "classifier" in name and param.requires_grad:
            if param.ndim >= 2:
                torch.nn.init.kaiming_normal_(param)
            else:
                torch.nn.init.zeros_(param)
            head_param_names.append(name)
    if head_param_names:
        print("Re-initialized head params:", head_param_names)

    return model
