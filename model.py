import torch
import torchvision.models as tv_models
import segmentation_models_pytorch as smp


# Defining custom segmentation model that supports Multi-Modal input (RGB + LiDAR)
class MultiModalDeepLabV3Plus(torch.nn.Module):
    def __init__(self, num_classes, return_features=False):
        super().__init__()
        self.return_features = return_features

        # Initializing the DeepLabV3+ model with 6 input channels (3 RGB + 3 LiDAR)
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=6,
            classes=num_classes,
        )
        # Manually loading pretrained ResNet50 weights
        self._load_pretrained_encoder_weights()

    def _load_pretrained_encoder_weights(self):
        # Loading standard pretrained ResNet50 from torchvision
        pretrained_resnet = tv_models.resnet50(pretrained=True)
        pretrained_dict = pretrained_resnet.state_dict()

        # Getting current encoder's weights
        model_dict = self.model.encoder.state_dict()

        # Filtering out the first convolutional layer (conv1), since it expects 3 channels
        filtered_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("conv1.")}

        # Updating encoder weights with compatible pretrained layers
        model_dict.update(filtered_dict)
        self.model.encoder.load_state_dict(model_dict)
        print("Loaded pretrained ResNet50 weights for encoder (excluding conv1)")

    def forward(self, x):
        # Returns only encoder features if feature extraction is requested
        if self.return_features:
            features = self.model.encoder(x)
            return features
        else:
            # Returns the full DeepLab V3+ output
            return self.model(x)

    def forward_features(self, x):
        # Explicit call for feature extraction (E.g in contrastive learning)
        return self.model.encoder(x)


def get_multimodal_deeplabv3plus(num_classes, return_features=False):
    return MultiModalDeepLabV3Plus(num_classes=num_classes, return_features=return_features)
























