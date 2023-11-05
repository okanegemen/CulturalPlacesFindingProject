import os
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
from DownloadTool import DownloadTool
import numpy as np

def take_Image_Path(dir:str=None):

    imageDirs = sorted(os.walk(dir))
    buf = []
    for i in range(1,len(imageDirs)):
        tupple_obj = imageDirs[i]

        for j in tupple_obj[2]:
            buf.append(f"{tupple_obj[0]}/{j}")

    return buf


def visualize(path_list:list):
    

    trashes = []

    for path in path_list:

        img = cv.imread(path)
        img = cv.resize(img,(600,600),interpolation =cv.INTER_AREA)
        cv.imshow("resim",img)
        YorN = input("iyi mi kötü mü seç aq iyiyse istediğin tuşa bas kötüyse n'ye sıkıldıysan break yaz  : ")
        cv.waitKey(1)
        cv.destroyAllWindows()
        
        if YorN == ("n" or "N"):
            trashes.append(path.split("/")[-1:])
        elif YorN=="break":
            break
    
    return trashes




# data = pd.read_csv("/Users/okanegemen/Desktop/BitirmeProjesi/newScrapedLinkCompleted3.csv")

# download = DownloadTool(sourceDataFrame = data,
#             logging_path="/Users/okanegemen/Desktop/BitirmeProjesi/dataScraping/logging/logging.log",
#             folderName_to_save="data4")




# download.download_img()

from utils import make_zip

from utils import LoadData



ld = LoadData()

ld.directoryToCsvFileWithPath("/Users/okanegemen/Desktop/BitirmeProjesi/dataScraping/data4").to_csv("withImages.csv",index=False)



