import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

# Load and preprocess the dataset
df = pd.read_csv("music_classification.csv")
df = df.drop(labels="beats", axis=1)
df['class_name'] = df['class_name'].astype('category')
df['class_label'] = df['class_name'].cat.codes
Genre_name = dict(zip(df.class_label.unique(), df.class_name.unique()))

X = df.iloc[:, 1:28]
y = df['class_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
svm = SVC(kernel='poly')
svm.fit(X_train_scaled, y_train)

logreg = LogisticRegression(solver='newton-cholesky', class_weight='balanced', max_iter=4000)
logreg.fit(X_train_scaled, y_train)

xgbc = xgb.XGBClassifier(n_estimators=100, random_state=3)
xgbc.fit(X_train_scaled, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Save models and scaler
with open('pickle/svm_model.pkl', 'wb') as file:
    pickle.dump(svm, file)

with open('pickle/logreg_model.pkl', 'wb') as file:
    pickle.dump(logreg, file)

with open('pickle/xgbc_model.pkl', 'wb') as file:
    pickle.dump(xgbc, file)

with open('pickle/knn_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

with open('pickle/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

with open('pickle/label_encoder.pkl', 'wb') as file:
    pickle.dump(Genre_name, file)
