import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('datasets/data_with_nans.csv')
data.head(3)


data.info()


data.drop(data.columns[0], axis=1, inplace=True)


data.isna().sum()


# data.groupby('Species').mean()

for column in data.columns[1:-1]:
    data[column].fillna(value=data[column].mean(), inplace=True)


data.isna().sum()


# for col in data.columns[1:-1]:
#     plt.figure(figsize=(8,6))
#     sns.scatterplot(x=data.Id, y=data[col], hue=data.Species)
#     plt.show()


cols = data.columns[1:-1]


species = data.Species.unique()


for c in cols:
    for s in species:
        df = data[data.Species == s]

        mean = df[c].mean()
        std = df[c].std()

        s3_max = mean + (std*3)
        s3_min = mean - (std*3)

        outlier = df[(df[c] > s3_max) | (df[c] < s3_min)]
        data.drop(index=outlier.index, axis=0, inplace=True)
    


data.shape


#for col in data.columns[1:-1]:
#    plt.figure(figsize=(8,6))
#    sns.scatterplot(x=data.Id, y=data[col], hue=data.Species)
#    plt.show()


data.head(3)


for c in cols:
    for s in species:
        df = data[data.Species == s]

        Q1 = df[c].quantile(0.25)
        Q3 = df[c].quantile(0.75)
        
        IQR = Q3-Q1
        step = IQR*1.5

        outlier = df[(df[c] > (Q3+step)) | (df[c] < (Q1-step))]
        data.drop(index=outlier.index, axis=0, inplace=True)
    


data


#for col in data.columns[1:-1]:
#    plt.figure(figsize=(8,6))
#    sns.scatterplot(x=data.Id, y=data[col], hue=data.Species)
#    plt.show()


data.drop('Id', axis=1, inplace=True)


data.to_csv('datasets/final_data.csv')



