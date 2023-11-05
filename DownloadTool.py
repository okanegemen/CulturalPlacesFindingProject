import pandas as pd
import urllib.request
from urllib.request import Request, urlopen
import base64
import logging
import os
import socket
from tqdm import tqdm

from time import strftime
from http.cookiejar import CookieJar


import time
from  datetime import datetime



class DownloadTool():
    def __init__(self) -> None:

        self.commonDir = None
        

    def check_for_b64(self,source):
        possible_header = source.split(',')[0]
        if possible_header.startswith('data') and ';base64' in possible_header:
            image_type = possible_header.replace('data:image/', '').replace(';base64', '')
            return image_type
        return False 

    def downloadWithUrlLib(self,url):
        req=urllib.request.Request(url, None, {'User-Agent': 'Mozilla/5.0 (X11; Linux i686; G518Rco3Yp0uLV40Lcc9hAzC1BOROTJADjicLjOmlr4=) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
                                                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                                                        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                                                        'Accept-Encoding': 'gzip, deflate, sdch',
                                                        'Accept-Language': 'tr-TR,tr;q=0.8',
                                                        'Connection': 'keep-alive'})

        cj = CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        response = opener.open(req)
        content = response.read()

        response.close()






        return content


    def checkAndCreate(self,dir):

        
        if os.path.exists(f"{dir}") is False:
                os.mkdir(f"{dir}")
        else:
            print("No such file directory Named -----> %s"%dir)


    def downloadImages(self,img_src,rootDir:str = "/Users/okanegemen/Desktop/BitirmeProjesi/dataScraping" , folderName:str = "data" ):


        self.commonDir = f"{rootDir}/{folderName}"




        self.checkAndCreate(self.commonDir)

        serie = img_src.groupby("PLACES")["PLACES"].count()
        names = serie.index.values
        

        for name in tqdm(names,desc="Image folders are creating",total=len(names),colour="blue"):
            placesDir = os.path.join(self.commonDir,name)

            self.checkAndCreate(placesDir)


        date_format = strftime('[%Y/%m/%d %T]')
        logging.basicConfig(filename='logging/logging.log', filemode='w', format='%(name)s | %(levelname)s | %(message)s | %(asctime)s',datefmt=date_format)

        count = 0
        
        for row in tqdm(img_src.itertuples( index = True , name = 'Pandas') , desc = "Images are downloading....",colour="red",total=img_src.shape[0]):
            is_b64 = self.check_for_b64(row.LINKS)


            if is_b64:
                
                content = base64.b64decode(row.LINKS.split(';base64')[1])
                try:
                    with open(f"{self.commonDir}/{row.PLACES}/{row.PLACES + str(count)}.png".format(content), 'wb') as f:
                        f.write(content)
                    f.close()
                    count+=1
                except Exception as e:
                    date = datetime.now() 

                    logging.error(''+row.PLACES+' | '+row.LINKS+' | '+str(e)+' | '+str(date)+'')
            # Else, if it is a direct URL, then perform urlretrieve to download the image
            else:
                try:

                    content = self.downloadWithUrlLib(row.LINKS)

                    with open(f"{self.commonDir}/{row.PLACES}/{row.PLACES + str(count)}.png".format(content),"wb") as f:
                        f.write(content)
                    f.close() 

                    count+=1

                    # urllib.request.urlretrieve(''+row.src_link+'', 'output/images/'+row.search_terms+'.png')
                except Exception as e: 
                    date = datetime.now() 
                    logging.error(''+row.PLACES+' | '+row.LINKS+' | '+str(e)+' | '+str(date)+'')