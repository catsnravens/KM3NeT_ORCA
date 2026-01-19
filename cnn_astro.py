import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout

from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from astropy.coordinates import CartesianRepresentation, SphericalRepresentation
from astropy import units as u

# ==============================================================================
# Our own function to determine loss
def cosine_distance_loss(y_true, y_pred):

    y_true_norm = tf.nn.l2_normalize(y_true, axis=1)
    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=1)

    # scalar vector (Cosine Similarity)
    # reslut : [-1 (opposite), 1 (same direction)]
    cosine_similarity = tf.reduce_sum(y_true_norm * y_pred_norm, axis=1)

    # Minimize the angle between the two vectors
    # angle 0  -> cos = 1. we want to maximize cosine similarity, cause we want to minimize the angle
    # Loss = 1 - cos
    return 1 - cosine_similarity

# ==============================================================================

# Main function to train and evaluate CNN model
def train_and_evaluate_cnn(file_name, class_label):

    # Load dataset
    print(f"\n{'='*40}")
    print(f"Training CNN for {class_label} - Loading data from {file_name}")
    print(f"\n{'='*40}")
    
    
    try:
        df = pd.read_hdf(file_name, key='y')
    except FileNotFoundError:
        return None

    initial_len = len(df)
    df = df.dropna()
    final_len = len(df)

    if final_len < initial_len:
        print(f"Rows removed : {initial_len - final_len}")

    if final_len == 0:
        return None


    #inputs and targets
    input_dir_cols = ['jmuon_dir_x', 'jmuon_dir_y', 'jmuon_dir_z', 'jmuon_likelihood']
    target_dir_cols = ['dir_x', 'dir_y', 'dir_z']

    # data matrices
    X_dir = df[input_dir_cols].values
    y_dir = df[target_dir_cols].values

    # Train-test split
    X_dir_train, X_dir_test, y_dir_train, y_dir_test = train_test_split(X_dir, y_dir, test_size=0.2, random_state=42)

    #PREPROCESSING

    # scaler for direction
    scaler_dir = StandardScaler()
    X_dir_train_scaled = scaler_dir.fit_transform(X_dir_train)
    X_dir_test_scaled = scaler_dir.transform(X_dir_test)

    #reshaping for CNN
    X_train_cnn = X_dir_train_scaled.reshape((X_dir_train_scaled.shape[0], 4, 1))
    X_test_cnn = X_dir_test_scaled.reshape((X_dir_test_scaled.shape[0], 4, 1))

    #actual model building

    model = tf.keras.Sequential()
    input_shape = (4,1)

    model.add(Input(shape=input_shape))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu'))

    # Flatten 
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(16, activation='relu')) #16 neurons in hidden layer > 32 > 64 (it was overtrained)
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='linear')) # 3 outputs for direction x,y,z

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #model.summary()

    # Training the model
    history = model.fit(x = X_train_cnn, y = y_dir_train, validation_data=(X_test_cnn, y_dir_test), epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluating the model
    y_pred = model.predict(X_test_cnn, verbose=0)

    # Calculate angle
    def get_angle(v1, v2):
        # we add 'epsilon' (1e-8) to avoid division by zero
        epsilon = 1e-8
        norm_v1 = np.linalg.norm(v1, axis=1, keepdims=True) + epsilon
        norm_v2 = np.linalg.norm(v2, axis=1, keepdims=True) + epsilon

        v1_n = v1 / norm_v1
        v2_n = v2 / norm_v2

        dot = np.sum(v1_n * v2_n, axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    angles = get_angle(y_dir_test, y_pred)
    # Remove NaN values if any
    angles = angles[~np.isnan(angles)]

    if len(angles) == 0:
        median_error = np.nan
    else:
        median_error = np.median(angles)

    print(f"Angle loss median: {median_error:.2f} degrees")
    return median_error, history


# ==============================================================================
#= we want to train and evaluate for all files =

files_map = {
    'Class A (Anti-Elec CC)': 'atm_neutrino_classA.h5',
    'Class B (Anti-Muon CC)': 'atm_neutrino_classB.h5',
    'Class C (Anti-Muon NC)': 'atm_neutrino_classC.h5',
    'Class D (Anti-Tau CC)':  'atm_neutrino_classD.h5',
    'Class E (Elec CC)':      'atm_neutrino_classE.h5',
    'Class F (Muon CC)':      'atm_neutrino_classF.h5',
    'Class G (Muon NC)':      'atm_neutrino_classG.h5',
    'Class H (Tau CC)':       'atm_neutrino_classH.h5',
}

results = {}
histories = {} #dict to store histories for each class

for label, filename in files_map.items():
    filepath = f"data/{filename}"
    result_tuple = train_and_evaluate_cnn(filepath, label)
    if result_tuple is not None:
        metric, hist = result_tuple
        results[label] = metric
        histories[label] = hist


# ==============================================================================
# Plotting results
if results:
    for cls, err in results.items():
        print(f"{cls}: {err:.2f}°")

    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    values = list(results.values())

    bars = plt.bar(names, values, color='darkred')

    plt.ylabel('Loss reconstruction angle median (degrees)')
    plt.title('CNN Direction Reconstruction Performance by Neutrino Class')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.1f}°', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('plots/cnn_direction_reconstruction_performance.png')
    plt.show()
else:
    print("Err2")

# ==============================================================================
# Plot training & validation loss for each class



if histories:
    for class_label, history_obj in histories.items():


        loss = history_obj.history['loss']
        val_loss = history_obj.history['val_loss']
        epochs_range = range(1, len(loss) + 1)


        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, loss, 'b-', linewidth=2, label='Training loss', color='rebeccapurple')
        plt.plot(epochs_range, val_loss, 'r-', linewidth=3, label='Validation loss', color='darkred')


        plt.title(f'Learning Curve: {class_label}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (Cosine Distance)')
        plt.legend(loc='upper right')
        plt.grid(True, which='both', linestyle='--', alpha=0.6)


        plt.savefig(f'plots/training_validation_loss_{class_label.replace(" ", "_")}.png')
        plt.show()
     
