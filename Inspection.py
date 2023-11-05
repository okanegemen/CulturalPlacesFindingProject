import pandas as pd
import numpy as np


data = pd.read_csv("/Users/okanegemen/Desktop/BitirmeProjesi/linksAndPlaces.csv")



serie  = data.groupby("PLACES")["PLACES"].count()
print(serie.index.values)

