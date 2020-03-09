## File for intermidiate testing

import pandas as pd

from data_loader import DataLoader
from preprocessor import Preprocessor
from CONSTANTS import *

a,b = DataLoader(SAMPLE_DATA_DIR).load()

aa = pd.DataFrame([['This is an amazing movie.', 1]], columns=['text', 'sentiment'])
Preprocessor().preprocess(a, stemming=True)
print(a)
