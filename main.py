import pandas as pd
import numpy as np

import os
os.environ['KERAS_BACKEND'] = 'theano' # Using 'theano' backend for keras

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, TimeDistributed, Activation

# Loading the data into a DataFrame
df = pd.read_csv("names.csv")

# Converting df column to list
names = list(df["name"][:])
names = [item.lower() for item in names]

#Creating vocab for Index to Char
itc = {key: value for key, value in enumerate(sorted(list(set("".join(names)))))}
itc[26] = "\n"

# Creating vocab for Char to Index
cti = {itc[key]: key for key in itc.keys()}

# Setting the maximum length of a name(CAN BE CHANGED)
max_len = 30

# Create a zeros array with necessary shape
data = np.zeros((len(names), max_len, len(itc.keys())))

# Fill the data array with correct encodings of the names
for i in range(data.shape[0]):
  error = False
  for j in range(data.shape[1]):
    try:
      data[i, j, cti[names[i][j]]] = 1
    except IndexError:
      data[i, j, cti["\n"]] = 1; error = True; break

  if not error: data[i, data.shape[1] - 1, cti["\n"]] = 1

# Reassigning the data to training samples X and targets Y
X, Y = np.zeros(data.shape), np.zeros(data.shape)
X[:, :, :], Y[:, :Y.shape[1] - 1, :] = data[:, :, :], data[:, 1:, :]

# Splitting the Data into training and evaluation sets
split_to_and_from = 374000

X_train, Y_train, X_test, Y_test = (X[:split_to_and_from, :, :], Y[:split_to_and_from, :], X[split_to_and_from:, :, :], Y[split_to_and_from:, :])

##### CREATING THE MODEL

# Hyper - params
time_steps = max_len
b_size = 2000
n_cells = 30
l_r = 0.01

print("Creating model...")
global model
model = Sequential([
    LSTM(n_cells, input_shape=(None, X.shape[2]), return_sequences=True),
    TimeDistributed(Dense(len(cti), activation="softmax"))
])
print("Model created!")

# Compiling the model
model.compile("rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

def pred(length):
	'''
	Generate a name with the model.
	'''
	X_pred = np.zeros((1, length, len(cti)))
	pred = [np.random.randint(len(cti) - 1)]

	gen = [itc[pred[-1]]]

	for i in range(length):
		X_pred[0, i, :][pred[-1]] = 1
		pred_arr = model.predict(X_pred[:, :, :], batch_size=1, verbose=0)
		pred_arr = pred_arr[0, i, :]
		p = np.argmax(pred_arr)

		pred.append(p)
		gen.append(itc[pred[-1]])

	return "".join(gen)


def train_model(num_epochs):
    print("Training the model now for {} epochs...".format(num_epochs))
    current_e = 1

    while current_e < num_epochs:
        # Fitting the model
        model.fit(x=X_train, y=Y_train, batch_size=b_size, epochs=1, verbose=1)

        # Generate a name of max_len
        print(pred(max_len).strip("\n"))
    print("Training complete!")


def save_my_model():
    # Saves weights and architecture
    model.save("my_keras_model.h5")


# Training the model
train_model(10)

# Save model
save_my_model()

# Evaluating the model
metrics = model.evaluate(x=X_test, y=Y_test, batch_size=100)

print("Model metrics: ")
for m_name, m in zip(model.metrics_names, metrics):
    print("{}: {}".format(m_name, m))
