import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense



# Load dataset
df = pd.read_hdf('data/atm_neutrino_classA.h5', 'y')

print(df.head())
print(df.columns)

#inputs

input_energy_cols = ['jmuon_E', 'jmuon_JENERGY_ENERGY', 'jmuon_JENERGY_CHI2', 'jmuon_JENERGY_NDF', 'jmuon_JGANDALF_NUMBER_OF_HITS', 'jmuon_JSHOWERFIT_ENERGY', 'jmuon_AASHOWERFIT_ENERGY']
input_dir_cols = ['jmuon_dir_x', 'jmuon_dir_y', 'jmuon_dir_z', 'jmuon_likelihood']

#targets

target_energy_col = 'energy'
target_dir_cols = ['dir_x', 'dir_y', 'dir_z']

# data matrices

X_energy = df[input_energy_cols].values
X_dir = df[input_dir_cols].values
y_energy = df[target_energy_col].values
y_dir = df[target_dir_cols].values

print("Input energy shape:", X_energy.shape)
print("Input direction shape:", y_dir.shape)


# Train-test split
X_en_train, X_en_test, X_dir_train, X_dir_test, y_en_train, y_en_test, y_dir_train, y_dir_test = train_test_split(X_energy, X_dir, y_energy, y_dir, test_size=0.2, random_state=42)

print("Training number of samples:", len(X_en_train))
print("Testing number of samples:", len(X_en_test))

#Preprocessing


# scaler for energy
scaler_energy = StandardScaler()
X_en_train_scaled = scaler_energy.fit_transform(X_en_train)
X_en_test_scaled = scaler_energy.transform(X_en_test)

# scaler for direction
scaler_dir = StandardScaler()
X_dir_train_scaled = scaler_dir.fit_transform(X_dir_train)
X_dir_test_scaled = scaler_dir.transform(X_dir_test)


print ("Statistics before scaling:")
print(f"X_en_train mean: {np.mean(X_en_train[:,0]):.2f}")
print("X_en_train std:", np.std(X_en_train[:,0]))
print("X_dir_train mean:", np.mean(X_dir_train[:,0]))
print("X_dir_train std:", np.std(X_dir_train[:,0]))

#reshaping for CNN
print("Shape before reshaping:", X_dir_train_scaled.shape)
X_train_cnn = X_dir_train_scaled.reshape((X_dir_train_scaled.shape[0], X_dir_train_scaled.shape[1], 1))
X_test_cnn = X_dir_test_scaled.reshape((X_dir_test_scaled.shape[0], X_dir_test_scaled.shape[1], 1))
print("Shape after reshaping for CNN:", X_train_cnn.shape)

#actual model building

model = tf.keras.Sequential()

input_shape = (4,1)

model.add(Input(shape=input_shape))
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'))

# Flatten 
model.add(tf.keras.layers.Flatten())
model.add(Dense(8, activation='relu')) #for now, later 32 
model.add(Dense(3, activation='linear')) # 3 outputs for direction x,y,z
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


history = model.fit(x = X_train_cnn, y = y_dir_train, validation_data=(X_test_cnn, y_dir_test), epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Plot training & validation loss values
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss -8')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.savefig('training_validation_loss8.png')
plt.show()
