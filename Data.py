import pandas as pd
from glob import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import  torchvision.transforms as T
import cv2 as cv


class PlacesData(Dataset):
    def __init__(self,rootDir,transforms,labels) :
        super().__init__()

        self.rd = rootDir
        self.transforms = transforms
        self.labels = labels

        img_folders = sorted(glob(f"{self.rd}/*"))
        self.img_list = []
        for img_fold in img_folders:
            self.img_list.extend(sorted(glob(f"{img_fold}/*")))

        

        for imgs in self.img_list:
            try:
                img = cv.imread(imgs)
                img = cv.resize(img,(224,224),interpolation=cv.INTER_AREA)
            except cv.error:
                self.img_list.remove(imgs)


        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):


        img = cv.imread(self.img_list[index])
        img = cv.resize(img,(224,224),interpolation=cv.INTER_AREA)
        transformed = self.transforms(img)
        label = self.labels[self.labels["PATHS"] == self.img_list[index]]["LABELS"].values[-1]
        
        return transformed,label






# print(data[data["PATHS"]=="/Users/okanegemen/CulturalPlacesFindingProject/dataScraping/dataLast/Akdamar_Kilisesi_Van/Akdamar_Kilisesi_Van1555.png"]["LABELS"].values[-1])

# print(data.shape)
















