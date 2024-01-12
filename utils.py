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
import faiss
from tqdm import tqdm

from torch.backends import mps
import statsmodels.stats.weightstats as st

from scipy import stats

import config as cfg
import PIL.Image as Image



warnings.filterwarnings("ignore")


class DataStuff():
    def __init__(self):
        pass
        
    def _getImagesPathsFromFolder(self,rootPathOfImagesFolders:str,save:bool = True, savedName:str="dataWithImages"):

        
        folders = sorted(os.walk(rootPathOfImagesFolders))

        dicti = {
            "NAMES" : [],
            "PATHS" : [],

        }
        count = 0

        for child in folders[1:]:
            for image in child[2]:
                
                dicti["NAMES"].append(str(child[0].split("/")[-1]))
                dicti["PATHS"].append(str(os.path.join(child[0],image)))

                
        
        df = pd.DataFrame(data=dicti,copy=True)
        

        uniques = list(df["NAMES"].unique())

        labels = [uniques.index(name) for name in df["NAMES"]]

        df["LABELS"] = labels

        cur_dir = os.getcwd()
        if save:

            df.to_csv(f"{cur_dir}/metaData/{savedName}.csv",index=False)

        return df
        

    def makeArray(self,rawData):

        return np.safe_eval(rawData)

    

    def _getImageList(self,dfPathOrDf):

        if type(dfPathOrDf) is str :

            data = pd.read_csv(dfPathOrDf)
            paths = data["PATHS"].values.tolist()
        else:
            paths = dfPathOrDf["PATHS"].values.tolist()
            return [dfPathOrDf,paths]

        return [data,paths]

    def createIndexFilePath(self,name:str = "indexedImagesFeaturesData"):

        curr_dir = os.getcwd()
        return os.path.join(curr_dir,"metaData",f"{name}.idx")

    def createPicklePath(self,name:str="featuresWithPaths"):
        curr_dir = os.getcwd()
        return os.path.join(curr_dir,"metaData",f"{name}.pkl")


        

class FeatureExtraction(nn.Module,DataStuff):
    def __init__(self):
        super().__init__()
       

        
    def _getModelAndFuse(self,model_names:list = ["resnext50_32x4d","vit_b_32"],pretrained:bool = True):

        model1 = models.get_model(name=model_names[0],pretrained = pretrained)
        model2 = models.get_model(name=model_names[1],pretrained = pretrained)

        feature_extract1 = [child for child in model1.children()]

        feature_extract2 = [child for child in model2.children()]

        self.modelList = [nn.Sequential(*feature_extract1[:-1]),nn.Sequential(*feature_extract2[:-1])]

        for idx,i in enumerate(model_names):
            if i.startswith("vit"):
                self.idx = idx
                self.extra = True
                self.modelList.append(models.get_model(name=model_names[idx],pretrained = pretrained))
            else:
                self.extra = False



        return self.modelList

    def _transformToTorchFormat(self,img):
        """
        this is the function which taked images and turned into torch format

        Args:
            img (np.array): It's type should be np.array type image 
            

        Returns:
            torch.tensor: The function is return torch.tensor type images
        """

        return cfg.TRANSFORMS(img)


    def _extract(self, img):


        """
        Do not forget if vit model you must take the feature of output[:,0,:]
        """


        if str(type(img)) != "<class 'torch.Tensor'>":
            x = self._transformToTorchFormat(img)
            x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        else:
            x = Variable(img.float(), requires_grad=False)

        

        
        x = x.to(cfg.DEVICE)
        torch.mps.synchronize()

    

        if self.extra:


            encoder = self.modelList[self.idx][1]
            encoder.to(cfg.DEVICE)
            self.modelList[2].to(cfg.DEVICE)
            pro = self.modelList[2]._process_input(x)
            n = pro.size(0)

            
            batch_class_token = self.modelList[2].class_token.expand(n,-1,-1)

            feature1 = torch.cat([batch_class_token, pro], dim=1)
            feature1 = encoder(feature1)
            feature1 = feature1[:,0]

            feature1 = feature1[:,:,None,None]
           
            torch.mps.empty_cache()
            self.modelList[1-self.idx].to(cfg.DEVICE)
            feature2 = self.modelList[1-self.idx].eval()(x)


       
        else:
            
            torch.mps.empty_cache()
            
            self.modelList[0].to(cfg.DEVICE)
            self.modelList[1].to(cfg.DEVICE)
            feature1 = self.modelList[0].eval()(x)
            feature2 = self.modelList[1].eval()(x)


        

        fusing = torch.cat([feature1,feature2],dim=1)

        
        features = fusing.detach().cpu().numpy().flatten()
        return features / np.linalg.norm(features)


    def _readImage(self,img):

        self.error = []
        try:

            readed_pil = Image.open(img)
            resized = readed_pil.resize((224,224))
            # readed = cv.imread(img)
            # readed = cv.cvtColor(readed,cv.COLOR_BGR2RGB)
            # resized = cv.resize(readed,(224,224),interpolation=cv.INTER_AREA)
            return resized
        except Exception as e:
            self.error.append(e)

            return None


    
    def createCityColumn(self,df):

        names = df["NAMES"].tolist()

        cities = [city.split("_")[-1] for city in names]

        df["CITIES"] = cities

        return df
         

        


    def _beginExtractFeatures(self,imgList,df):
       
        features = []

        ds = DataStuff()


        for img in tqdm(imgList,desc= "Feature Extraction and indexing is begined",total=len(imgList),colour="red"):


            if self._readImage(img) is not None:
                features.append(self._extract(self._readImage(img)))

            else:
                features.append(None)
                print(f"Image is not recognized at file '{img}' the error is {self.error[-1]} ")


        df["FEATURES"] = features
        df = df.dropna().reset_index(drop = True)


        df = self.createCityColumn(df)

        df.to_pickle(ds.createPicklePath())

        return df

    def _indexing(self,df:pd.DataFrame):

        assert len(df) != 0 , "There is no element in DataFrame"

        path = super().createIndexFilePath("indexedImagesFeaturesData")

        features = df["FEATURES"]
        dim = len(features[0])
        self.dim = dim
        idx = faiss.IndexFlatL2(dim)
        
        featureMatrix = np.vstack(features.values).astype(np.float32)

        idx.add(featureMatrix)


        faiss.write_index(idx,path)

        return path

    def _indexAllData(self):

        data,img_list =  super()._getImageList(cfg.IMAGES_PATH_DF)
        df = self._beginExtractFeatures(imgList=img_list,df=data)
        path = self._indexing(df)
        print(f"Indexing is completed and './...idx' file is saved at '{path}'!!!")




class SearchByIndexFile(FeatureExtraction):
    def __init__(self):
        super().__init__()
        


    def _searchByIndex(self,extracted,nRetrive,df):

        self.extracted = extracted
        self.n = nRetrive

        path = super().createIndexFilePath()

        index = faiss.read_index(path)
        
        dist,idx = index.search(np.array([self.extracted]).astype(np.float32),self.n)

        dictionary = {
            "idx":list(idx[0]),
            "paths": df.loc[idx[0]]["PATHS"].tolist(),
            "labels":df.loc[idx[0]]["LABELS"].tolist(),
            "distances": np.squeeze(dist).tolist(),
            "places" : [" ".join(place.split("_")[:-1]) for place in df.loc[idx[0]]["NAMES"].tolist()],
            "cities": df.loc[idx[0]]["CITIES"].tolist()

        }


        meanOfDistances = np.mean(dictionary["distances"])
        
    
        if meanOfDistances > 0.73:

            return f"Please upload another image"

        mode=stats.mode(dictionary["labels"])

        
        mode = dictionary["foundedImage"]=(*mode[0],*mode[1])

        return dictionary

    def _extractQuery(self,query,modelList):

        if type(query) is str:

            img = super()._readImage(query)

        else:
            img = query
    
        super()._getModelAndFuse(model_names=modelList)

        extracted = super()._extract(img)

        return extracted



    
def getMetadata(name:str = None):
    """
    The function will return file data about name

    Args:
        name (str): Full of file_name because extantion is important just specify like 'name.csv or name.pkl'

    Returns:
        pd.DataFrame: Function will return  pandas dataframe type data 
    """

    assert type(name) is str , "Please specify string format path"

    cur_dir = os.getcwd()

    full_path = os.path.join(cur_dir,"metaData",name)

    if name.endswith(".csv"):
        data = pd.read_csv(full_path)

    elif name.endswith(".pkl"):
        data = pd.read_pickle(full_path)

    else:

        print("the Data file extantion is not supported.Please give '.pkl' or '.csv' type file !!!!!")

    return data




