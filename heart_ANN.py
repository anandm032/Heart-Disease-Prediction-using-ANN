import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

df=pd.read_csv("C:/Users/anand/Downloads/heart.csv")
print(df.head(5))

print(df.shape)
print(df.info())
print(df.describe())

print(df.isnull().sum())    


X = df.drop("target", axis=1)

y = df["target"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()

model.add(Dense(units=16,activation='relu',input_dim=13))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)

predictions = model.predict(X_test)

print(predictions[:5])

predictions = (predictions > 0.5)

print(predictions[:5])

predictions = predictions.astype(int)

print(predictions[:5])


print("Predicted:", predictions[:10].flatten())
print("Actual   :", y_test[:10].values)


print(history.history.keys())

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()





plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)
