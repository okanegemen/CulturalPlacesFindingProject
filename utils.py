import pandas as pd
import numpy as np 

import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as T 
import warnings
from torch.autograd import Variable
import os 
import cv2 as cv


warnings.filterwarnings("ignore")




class FeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()
       

    def _getModel(self,model_name,pretrained):

        model = models.get_model(name=model_name,pretrained = pretrained)

        feature_extract = [child for child in model.children()]

        return nn.Sequential(*feature_extract[:-1])

    def _transformToTorchFormat(self,img):
        """
        this is the function which taked images and turned into torch format

        Args:
            img (np.array): It's type should be np.array type image 
            

        Returns:
            torch.tensor: The function is return torch.tensor type images
        """
        transforms = T.Compose([

            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406] , 
                        std = [0.229, 0.224, 0.225])

        ])



        return transforms(img)


    def _extract(self, img,model):

        x = self.transformToTorchFormat(img)
        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

        feature = model(x)

        
        feature = feature.detach().cpu().numpy().flatten()
        return feature / np.linalg.norm(feature)

    def _beginExtractFeatures(self,imgList,df,model):
       
        features = []

        for img in imgList:
            try:
                readed = cv.imread(img)
                readed = cv.cvtColor(readed,cv.COLOR_BGR2RGB)
                resized = cv.resize(readed,(224,224),interpolation=cv.INTER_AREA)
                features.append(self._extract(resized,model))

            except cv.error as e:
                features.append(None)
                print(f"Image is not recognized at file '{img}' the error is {e}")
        df["FEATURES"] = features
        df = df.dropna().reset_index(drop = True)

        cur_dir = os.getcwd()

        df.to_pickle(f"{cur_dir}/featuresWithPaths.pkl")

        return df





class DataStuff():
    def __init__(self):
        pass
        
    def getImagesPathsFromFolder(self,rootPathOfImagesFolders:str,save:bool = True, savedName:str="dataWithImages"):

        
        folders = sorted(os.walk(rootPathOfImagesFolders))

        dicti = {
            "NAMES" : [],
            "PATHS" : [],
            "PIXELS":[]

        }

        for child in folders[1:]:
            for image in child[2]:
                
                    
                    dicti["NAMES"].append(str(child[0].split("/")[-1]))
                    dicti["PATHS"].append(str(os.path.join(child[0],image)))

                
        
        df = pd.DataFrame(data=dicti,copy=True)

        uniques = list(df["NAMES"].unique())

        labels = [uniques.index(name) for name in df["NAMES"]]

        df["LABELS"] = labels
        if save:

            df.to_csv(f"{savedName}.csv",index=False)



        
        return dicti
        

    def makeArray(self,rawData):

        return np.safe_eval(rawData)



    def getImageList(self,dfPath):
        data = pd.read_csv(dfPath)
        paths = data["PATHS"].values.tolist()

        return [data,paths]

         





   


