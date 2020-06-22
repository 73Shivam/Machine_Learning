import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# split dataset
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size =.2, random_state=0)