import pandas as pd
import numpy as np 
import torch
import torchvision
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as T 
import torch.functional as F

import os 
import shutil 
import pyspark as spark

from pyspark.sql.types import *

from pyspark.sql import SparkSession
from pyspark.sql.functions import *



class DataStuff():
    def __init__(self) -> None:
        
        pass
    def getFoldersAndCreateCsvWithImages(self,rootPath):
        pass

    def transformToTorchFormat(self,df):
        pass

    def preprocessing(self,df):
        pass

    def createDataLoader(self,preprocessedData):
        pass

    


