import torchvision
import torch

def anchorgen():
    anchor_sizes=((32, 64, 128, 256),),
    aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)

class FasterRCNN(torch.nn.Module):
    def __init__(self, num_classes, backbone_name='resnet'):
        super(FasterRCNN, self).__init__()

        if backbone_name.lower() == 'resnet':
            backbone = torchvision.models.resnet50(pretrained=True)
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
            backbone.out_channels = 2048
        elif backbone_name.lower() == 'vgg':
            backbone = torchvision.models.vgg16(pretrained=True).features
            backbone.out_channels = 512
        else:
            raise ValueError("Invalid backbone name.")
        
        self.model = torchvision.models.detection.FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchorgen(),
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        
    
    def forward(self, images, targets):
        return self.model(images, targets)