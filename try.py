import pandas as pd 

import numpy as np



import config as cfg
import requests
imageFile = {"image" : open("/Users/okanegemen/CulturalPlacesFindingProject/WhatsApp Image 2024-01-10 at 18.41.01.jpeg","rb")}

r = requests.post(f"http://{cfg.HOST}:{cfg.PORT}/postData", files=imageFile)

print(r.content)

getted = requests.get(f"http://{cfg.HOST}:{cfg.PORT}/getPreds")

print(getted.json())


# data.to_pickle("/Users/okanegemen/Desktop/CulturalPlacesFindingProject/metaData/featuresWithPaths.pkl")

