import torch

import torchvision.models as models
import torchvision.transforms as T


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DEVICEWIN = "cuda:0" if torch.cuda.is_available() else "cpu"

MODELS = ["resnet50","efficientnet_v2_s"]

ROOT = "root_dir/"
METADATAROOT = "metadata/"

DATASET = f"{ROOT}/dataLast" #dataLast is dataset Folder like dataLast/imagesFolders

PLACESPATH = f"{METADATAROOT}/CitiesAndPlaces.csv"

LINKSPATH = f"{METADATAROOT}/linksAndPlaces.csv"

IMAGES_PATH_DF =f"{METADATAROOT}/dataWithImages.csv"

FEATURES = f"{METADATAROOT}/featuresWithPaths.pkl"

FEATURES_INDEXES_L2 = f"{METADATAROOT}/indexedImagesFeaturesData.idx"

WILL_RETURN_IMAGE_COUNT = 5

HOST = '0.0.0.0'

PORT = 5008

TRANSFORMS = T.Compose([

                T.ToTensor(),
                T.Normalize(mean = [0.485, 0.456, 0.406] , 
                            std = [0.229, 0.224, 0.225])

        ])
