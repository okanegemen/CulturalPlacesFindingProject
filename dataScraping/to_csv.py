
from  historicalPlaces import *
url = "https://www.enuygun.com/bilgi/turkiye-deki-en-populer-tarihi-yerler/"


thp = TakeHistoricalPlaces(url)

columns = ["Places","Cities"]

dictionary = {"data-testid" : "content-container-wrapper",
              "class" : "styled__BoxStyled-sc-ktges0-0 cqRCFx"}

listOfDivs = thp.takeNames("div",dictionary)

df = takeNamesAndCreateDf(listOfDivs,columns)

df.to_csv("scrapedData.csv",index=False)




