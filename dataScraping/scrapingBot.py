from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


import numpy as np 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import pandas as pd
import time

import os

from  selenium.common.exceptions import TimeoutException


class takeSource(object):

    def __init__(self):
       
        self.links = []
        self.Places = []
        self.Cities = []

    

    def check_exists_by_xpath(self,driver,xpath):
        try:
            driver.find_element(By.XPATH,value=xpath)
        except :
            return False
        return True


    def search_google(self,keyword,beginFrom,img_count):

        search_url = f"https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&q={keyword}"
        browser = webdriver.Chrome()
        
        browser.get(search_url)

        Places = "_".join(keyword.split(" "))

        #img_box = browser.find_element(by = By.XPATH ,value=f'//*[@id="islrg"]/div[1]/div[{str(2)}]/a[1]/div[1]/img')
        
        count = 0
        for i in tqdm(range(img_count),desc=f"Link Searching is beginned for {keyword}",colour="green",total=img_count):


            if self.check_exists_by_xpath(browser,f'//*[@id="islrg"]/div[1]/div[{str(beginFrom+i+1+count)}]/a[1]/div[1]/img'):


                try:

                    element = WebDriverWait(browser,20).until(EC.element_to_be_clickable((
                        By.XPATH, 
                        f'//*[@id="islrg"]/div[1]/div[{str(beginFrom+i+1+count)}]/a[1]/div[1]/img')))
                        
                    element.click()


                

                    fir_img = WebDriverWait(element,20).until(EC.element_to_be_clickable((
                    By.XPATH,
                    "//*[@id='Sva75c']/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div[3]/div[1]/a/img[1]")))



                    img_src = fir_img.get_attribute('src')

                    self.links.append(img_src)

                    self.Places.append(Places)

                    self.Cities.append(keyword.split(" ")[-1])

                except :
                    count+=1
                    continue
            else:

                count +=1

                try:

                    element = WebDriverWait(browser,20).until(EC.element_to_be_clickable((
                        By.XPATH, 
                        f'//*[@id="islrg"]/div[1]/div[{str(beginFrom+i+1+count)}]/a[1]/div[1]/img')))
                        
                    element.click()


                    fir_img = WebDriverWait(element,20).until(EC.element_to_be_clickable((
                    By.XPATH,
                    "//*[@id='Sva75c']/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div[3]/div[1]/a/img[1]")))


                    img_src = fir_img.get_attribute('src')




                    self.links.append(img_src)

                    self.Places.append(Places)

                    self.Cities.append(keyword.split(" ")[-1])
                except:

                    count+=1
                    continue

            # XPath of the image display 
        
        return self.links


    def turnToDfAndSaveLinksPlacesAndCity(self,name : str= None, save:bool =False):


        assert( type(name) is str and save is True) or (type(name) is None and save is False), "if you specify save as True you should specify name also"


        data = pd.DataFrame()

        data["PLACES"] = self.Places
        data["LINKS"] = self.links
        data["CITIES"] = self.Cities

        if save:

            curDir = os.getcwd()

            data.to_csv(f"{curDir}/metaData/{name}",index=False)

            print(f"DataFrame is saved at located -------->  {curDir}/{name}")

        else: return data




# Creating header for file containing image source link 
# with open("output/links/img_src_links.csv", "w") as outfile:
#     outfile.write("search_terms|src_link\n")

# # Loops through the list of search input
# for keyword in keywords['scientific_names']:
#     try:
#         link = search_google(keyword)
#         keyword = keyword.replace(" ", "_")
#         with open("output/links/img_src_links.csv", "a") as outfile:
#             outfile.write(f"{keyword}|{link}\n")
#     except Exception as e: 
#         print(e)
