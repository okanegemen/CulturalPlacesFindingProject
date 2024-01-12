from utils import *
import config as cfg
def dataOperation(model_list,createCsv = False):

    if createCsv:


        ds = DataStuff()

        df = ds._getImagesPathsFromFolder("/Users/okanegemen/CulturalPlacesFindingProject/dataScraping/dataLast")

    fe = FeatureExtraction()

    fe._getModelAndFuse(model_list)
    fe._indexAllData()


def search(modelList,queryImage,nImg):

    search = SearchByIndexFile()
    data = pd.read_pickle(cfg.FEATURES)
    extracted = search._extractQuery(queryImage,modelList)
    dicti = search._searchByIndex(extracted,nImg,data)
    return dicti




if __name__ == "__main__":

    #dataOperation(cfg.MODELS)
    dicti = search(cfg.MODELS,"/Users/okanegemen/CulturalPlacesFindingProject/TestImages/ef7ab834-b9fa-40fd-ad23-6d084d9d947f2602005709849348555.jpg",cfg.WILL_RETURN_IMAGE_COUNT)
    print(dicti)
    df = pd.read_csv("/Users/okanegemen/CulturalPlacesFindingProject/metaData/dataWithImages.csv")

    if type(dicti) is not str:

        print(df[df["LABELS"]==dicti["foundedImage"][0]].head())

