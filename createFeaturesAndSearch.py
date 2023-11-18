from utils import *

def dataOperation():

    fe = FeatureExtraction()

    fe._getModelAndFuse()
    fe._indexAllData()


def search(queryImage,nImg):

    search = SearchByIndexFile()

    cur_dir = os.getcwd()

    data = pd.read_pickle(f"{cur_dir}/metaData/featuresWithPaths.pkl")

    extracted = search._extractQuery(queryImage)


    dicti = search._searchByIndex(extracted,nImg,data)

    print(dicti)




if __name__ == "__main__":

    # dataOperation()

    search("/Users/okanegemen/Desktop/CulturalPlacesFindingProject/Figure_4.png",5)



