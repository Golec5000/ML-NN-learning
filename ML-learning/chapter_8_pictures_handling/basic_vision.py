import cv2
import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models

image_bgr = cv2.imread("wombat.jpeg", cv2.IMREAD_COLOR)

convert_tensor = transforms.ToTensor()
pytorch_image = convert_tensor(np.array(image_bgr))

model = models.resnet18(pretrained=True)

layer = model._modules.get("avgpool")

model.eval()

with torch.no_grad():
    emmbeddings = model(pytorch_image.unsqueeze(0))

print(emmbeddings.shape)
