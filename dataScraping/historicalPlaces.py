from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np 
import PIL
import os 



class TakeHistoricalPlaces:
    def __init__(self,url):
        self.url = url

    
    def getSource(self):

        response = requests.get(self.url)
        return response.content
    
    def takeNames(self,willFindHtmlBlock,dictiToSpesificClassFinding):

        html = self.getSource()
        soup = BeautifulSoup(html,"html.parser")
        
        liste = soup.find_all(willFindHtmlBlock,dictiToSpesificClassFinding)
        return liste



url = "https://www.enuygun.com/bilgi/turkiye-deki-en-populer-tarihi-yerler/"
obj = TakeHistoricalPlaces(url)

dictionary = {"data-testid" : "content-container-wrapper",
              "class" : "styled__BoxStyled-sc-ktges0-0 cqRCFx"}

listOfDivs = obj.takeNames("div",dictionary)





def takeNamesAndCreateDf(listDiv:list,columns:list = None):
    namesAndCity = [div.find_all("h3") for div in listDiv if len(div.find_all("h3"))>0]

    array = np.array(namesAndCity)
    array = array[:].all(axis=0)


    dicti = {
        "PLACES" : [],
        "CITIES" : []
    }




    for txt in array:

        dicti["PLACES"].append(txt.text.strip().split(",")[0].split("-")[1][1:])
        dicti["CITIES"].append(txt.text.strip().split(",")[1][1:])

    df = pd.DataFrame(dicti)


    
    return df



takeNamesAndCreateDf(listOfDivs).to_csv("CitiesAndPlaces.csv",index=False)

    




















    


    
    
        

