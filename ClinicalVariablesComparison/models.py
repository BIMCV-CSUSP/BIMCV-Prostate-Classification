import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBN


class EfficientNet_pretrained(nn.Module):
    """
    EfficientNet_pretrained: A class that utilizes a pretrained EfficientNet model 
    for 3D image classification, designed for grayscale 3D volumes (e.g., structural brain MRIs).

    The model uses an EfficientNet backbone for feature extraction and classification.

    Args:
        model_name (str): Model version/name to use from the EfficientNet variants. Defaults to "efficientnet-b0".
        n_classes (int): Number of output classes. Defaults to 2.
        in_channels_eff (int): Number of input channels for EfficientNet. Defaults to 1.
        pretrained_weights_path (str): Path to the pretrained weights of EfficientNet (Optional). Defaults to None.

    Forward Return:
        x (Tensor): The classification logits.

    Example:
        model = EfficientNet_pretrained(model_name='efficientnet-b0', n_classes=2, pretrained_weights_path=None)
    """

    def __init__(
        self,
        model_name: str = "efficientnet-b0",
        n_classes: int = 2,
        in_channels_eff: int = 1,
        pretrained_weights_path: str = None,
    ):
        """
        Initialize the EfficientNet_pretrained model with the given parameters.

        Args:
            model_name (str): Model version/name to use from the EfficientNet variants.
            n_classes (int): Number of output classes.
            in_channels_eff (int): Number of input channels for EfficientNet.
            pretrained_weights_path (str): Path to the pretrained weights of EfficientNet (Optional).
        """
        super(EfficientNet_pretrained, self).__init__()

        # Instantiate the EfficientNet model
        EfficientNet = EfficientNetBN(
            model_name=model_name,
            pretrained=False,
            progress=False,
            spatial_dims=3,
            in_channels=in_channels_eff,
            num_classes=n_classes,
        )

        # Load pretrained weights into EfficientNet if provided
        if pretrained_weights_path:
            EfficientNet.load_state_dict(torch.load(pretrained_weights_path)["state_dict"])

        self.model = EfficientNet

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (Tensor): Input image tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            x (Tensor): Classification logits of shape (batch_size, n_classes).
        """
        
        x = self.model(x)

        return x



def test():
    model = EfficientNet_pretrained(
        model_name="efficientnet-b7",
        n_classes=2,
        in_channels_eff=3,
        pretrained_weights_path=None,
    )
    # print(model)
    input = torch.randn(3, 3, 128, 128, 32)
    out = model(input)
    print(out.shape)
    print(out)
    # print(f"For input {input.size()}, output is {out.size()}")


if __name__ == "__main__":
    test()