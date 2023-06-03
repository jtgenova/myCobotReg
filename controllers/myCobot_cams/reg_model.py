import torch
import torch.nn as nn
from torchvision import transforms

class DictModel(nn.Module):
    def __init__(self, num_classes):
        super(DictModel, self).__init__()
        # used to transform images to tensors then resize to 512x512
        self.transforms = torch.nn.Sequential(transforms.Resize((512,512)))
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # layer to convert 4 channel image to 3 channel image
        self.rgb_layer = nn.Conv2d(4,3,1)

        # feature extractor resnet
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        
        # joint angle embedding
        self.joint_embedding = nn.Linear(num_classes, 1000)

        # define layers
        self.fc1 = nn.Linear(3000, num_classes)

        self.act = nn.LeakyReLU()

    def forward(self, x):
        top_img = x["Top Image"]
        side_img = x["Side Image"]
        joint_angles = x["Current joint Angles"]

        #apply transforms to images
        # top_img = transforms.ToPILImage()(top_img)
        # side_img = transforms.ToPILImage()(side_img)
        top_img = self.transforms(top_img)
        side_img = self.transforms(side_img)

        # top_img = self.convert(top_img)
        # side_img = self.convert(side_img)
        joint_angles = torch.Tensor(joint_angles)

        # send all to device
        top_img = top_img.to(self.DEVICE)
        side_img = side_img.to(self.DEVICE)
        joint_angles = joint_angles.to(self.DEVICE)
        
        # convert images to have 3 channels
        top_img = self.rgb_layer(top_img)
        side_img = self.rgb_layer(side_img)

        # extract features from images
        top_img = self.feature_extractor(top_img)
        side_img = self.feature_extractor(side_img)

        # embed joint angles
        joint_angles = self.joint_embedding(joint_angles)

        # concatenate features
        big_vec = torch.cat((top_img, side_img, joint_angles), dim=1) # makes 3000 dim vector

        # pass through final layer
        big_vec = self.act(big_vec)
        x = self.fc1(big_vec)

        return x