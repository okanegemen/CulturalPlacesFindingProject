from scrapingBot import *

from DownloadTool import *


keys = pd.read_csv("/Users/okanegemen/CulturalPlacesFindingProject/metaData/CitiesAndPlaces.csv")

places = keys["PLACES"]

cities = keys["CITIES"]




ts = takeSource()

for idx in range(len(places)):
    ts.search_google(f"{places[idx]} {cities[idx]}",0,40)



ts.turnToDfAndSaveLinksPlacesAndCity("linksAndPlaces.csv",True) 

dt = DownloadTool()

dt.downloadImages(img_src = pd.read_csv("/Users/okanegemen/CulturalPlacesFindingProject/metaData/linksAndPlaces.csv"),folderName="dataLast")






