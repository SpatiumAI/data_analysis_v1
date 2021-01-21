import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost as xgb


data = pd.read_csv('datasets/final_data.csv')


data.drop('Unnamed: 0', axis=1, inplace=True)


data.head(3)


le = LabelEncoder()


data.Species = le.fit_transform(data.Species)


data.head(3)


X_train, X_test, y_train, y_test = train_test_split(data.iloc[ : , :-1], data.iloc[ : , -1], test_size = 0.2, random_state = 42)


y_train.value_counts()


y_test.value_counts()


xgb_cls = xgb.XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False)


xgb_cls.fit(X_train, y_train)


preds = xgb_cls.predict(X_test)


accuracy_score(y_test, preds)


confusion_matrix(y_test, preds)



