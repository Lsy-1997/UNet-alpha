from torch import nn
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class FCN_resnet50(nn.Module):
    def __init__(self, n_channels, n_classes) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # weights = FCN_ResNet50_Weights.DEFAULT
        self.model = fcn_resnet50(weights=None, progress=False, num_classes=self.n_classes)

    def forward(self, x):
        return self.model(x)['out']
