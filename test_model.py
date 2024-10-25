import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import torch 

import cv2

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from PIL import Image
from transformers import AutoModelForImageClassification

class Single_model(LightningDataModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self.cfg = self.model.config
        self.transforms = self.make_transforms(data_aug=True)
    
    def make_transforms(self, data_aug: bool):
        augments = []
        if data_aug:
            aug = self.cfg.aug
            augments = [
                A.RandomResizedCrop(
                    self.cfg.image_size[0],
                    self.cfg.image_size[1],
                    scale=(aug.crop_scale, 1.0),
                    ratio=(aug.crop_l, aug.crop_r),
                ),]
        return A.Compose(augments)
    
    def transform(self, img):
        # img in bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb, self.cfg.image_size, interpolation=cv2.INTER_CUBIC)
        return self.transforms(image=image)["image"]

    def forward(self, img):
        if self.transforms is None:
            return self.model(img)
        else:
            image = self.transform(img)
            image = torch.Tensor(image).transpose(2, 0).unsqueeze(0)
            return self.model(image)

if __name__ == "__main__":
    cetacean_classifier = AutoModelForImageClassification.from_pretrained("Saving-Willy/cetacean-classifier", trust_remote_code=True)
    model = Single_model(cetacean_classifier)
    img = Image.open('tail.jpg')
    
    out = model(img)
    print(out)