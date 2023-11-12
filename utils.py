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


        










class FeatureExtraction(nn.Module):
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



        return self.modelList

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


    def _extract(self, img):


        """
        Do not forget if vit model you must take the feature of output[:,0,:]
        """

        
        device = "mps" if torch.backends.mps.is_available() else "cpu"

       
        x = self._transformToTorchFormat(img)

        x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)

        x = x.to(device)

    

        if self.extra:


            encoder = self.modelList[self.idx][1]
            encoder.to(device)
            self.modelList[2].to(device)
            pro = self.modelList[2]._process_input(x)
            n = pro.size(0)

            
            batch_class_token = self.modelList[2].class_token.expand(n,-1,-1)

            feature1 = torch.cat([batch_class_token, pro], dim=1)
            feature1 = encoder(feature1)
            feature1 = feature1[:,0]

            feature1 = feature1[:,:,None,None]

            self.modelList[1-self.idx].to(device)
            feature2 = self.modelList[1-self.idx].eval()(x)



            
        else:

            self.modelList[0].to(device)
            self.modelList[1].to(device)
            feature1 = self.modelList[0].eval()(x)
            feature2 = self.modelList[1].eval()(x)


        

        fusing = torch.cat([feature1,feature2],dim=1)

        
        features = fusing.detach().cpu().numpy().flatten()
        return features / np.linalg.norm(features)


    def _readImage(self,img):

        self.error = []
        try:


            readed = cv.imread(img)
            readed = cv.cvtColor(readed,cv.COLOR_BGR2RGB)
            resized = cv.resize(readed,(224,224),interpolation=cv.INTER_AREA)
            return resized
        except cv.error as e:
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

        cur_dir = os.getcwd()

        df = self.createCityColumn(df)

        df.to_pickle(ds.createPicklePath())

        return df

    def _indexing(self,df:pd.DataFrame):

        assert len(df) != 0 , "There is no element in DataFrame"

        path = DataStuff().createIndexFilePath("indexedImagesFeaturesData")

        features = df["FEATURES"]
        dim = len(features[0])
        self.dim = dim
        idx = faiss.IndexFlatL2(dim)
        
        featureMatrix = np.vstack(features.values).astype(np.float32)

        idx.add(featureMatrix)


        faiss.write_index(idx,path)

        return path

    def _indexAllData(self):

        data,img_list =  DataStuff()._getImageList("/Users/okanegemen/Desktop/CulturalPlacesFindingProject/dataWithImages.csv")
        df = self._beginExtractFeatures(imgList=img_list,df=data)
        path = self._indexing(df)
        print(f"Indexing is completed and './...idx' file is saved at '{path}'!!!")




class SearchByIndexFile(FeatureExtraction):
    def __init__(self):
        super().__init__()
        


    def _searchByIndex(self,extracted,nRetrive,df):

        self.extracted = extracted
        self.n = nRetrive

        path = DataStuff().createIndexFilePath()

        index = faiss.read_index(path)

        dist,idx = index.search(np.array([self.extracted]).astype(np.float32),self.n)

        dictionary = {
            "idx":idx[0],
            "paths": df.loc[idx[0]]["PATHS"].tolist(),
            "labels":df.loc[idx[0]]["LABELS"].tolist(),
            "distances": np.squeeze(dist).tolist(),

            "cities": df.loc[idx[0]]["CITIES"].tolist()

        }




        meanOfDistances = np.mean(dictionary["distances"])
        
    
        if meanOfDistances > 0.65:

            return f"Query Image is not found in Places "

        mode=stats.mode(dictionary["labels"])

        
        mode = dictionary["foundedImage"]=(*mode[0],*mode[1])

        return dictionary

    def _extractQuery(self,query):

        if type(query) is str:

            img = super()._readImage(query)

        
    
        super()._getModelAndFuse()

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




# search = SearchByIndexFile()

# data = pd.read_pickle("/Users/okanegemen/Desktop/CulturalPlacesFindingProject/metaData/featuresWithPaths.pkl")

# extracted = search._extractQuery("/Users/okanegemen/Desktop/CulturalPlacesFindingProject/dataLast/Ulu_Camii_Bursa/Ulu_Camii_Bursa3006.png")


# dicti = search._searchByIndex(extracted,5,data)

# print(dicti)

           