from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score,confusion_matrix
from Data import *
from utils import *
from createFeaturesAndSearch import *

from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np









def test(dataLoader):
    predicted_list = []
    true_label_list = []
    for i,(img,labels) in tqdm(enumerate(list(dataLoader)[0:3226:5]),total=646,colour="blue"):

        dicti = search(["resnet50","efficientnet_v2_s"],img,5)
        if type(dicti) is str:
            predicted_list.append(labels.item()-1)
        else:
            predicted_list.append(dicti["foundedImage"][0])
        true_label_list.append(labels.item())
    

    return true_label_list,predicted_list


    








class Metrics():
    def __init__(self) -> None:
        pass

    def recall(self,true_label_list,predicted_list):

        return recall_score(true_label_list,predicted_list,avarage = "weighted")

    def precision(self,true_label_list,predicted_list):

        return precision_score(true_label_list,predicted_list,avarage = "weighted")
        
    def f1_score(self,true_label_list,predicted_list):

        return f1_score(true_label_list,predicted_list,avarage = "weighted")

    def accuracy(self,true_label_list,predicted_list):

        return accuracy_score(true_label_list,predicted_list)

if __name__=="__main__":

    transforms = T.Compose([

            T.ToTensor(),
            T.Normalize(mean = [0.485, 0.456, 0.406] , 
                        std = [0.229, 0.224, 0.225])

        ])
        
    dataset = PlacesData("/Users/okanegemen/CulturalPlacesFindingProject/dataScraping/dataLast",transforms,pd.read_csv("/Users/okanegemen/CulturalPlacesFindingProject/metaData/dataWithImages.csv"))

    dataLoader = DataLoader(dataset)

    true_label_list,predicted_list = test(dataLoader)

    metrics = Metrics()

    accuracy = metrics.accuracy(true_label_list,predicted_list)

    precision = metrics.precision(true_label_list,predicted_list)

    recall = metrics.recall(true_label_list,predicted_list)

    f1_value = metrics.f1_score(true_label_list,predicted_list)



    




