import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

with open("soft_music_dataset.json", "r") as f:
    dataset = json.load(f)

sequences = []
for entry in dataset:
    notes = entry["notes"]
    sequences.append([[note[2], note[1] - note[0], note[3]] for note in notes])

def normalize_data(data):
    min_vals, max_vals = np.min(data, axis=0), np.max(data, axis=0)
    return (data - min_vals) / np.maximum(max_vals - min_vals, 1e-8), (min_vals, max_vals)

all_notes = [note for seq in sequences for note in seq]
normalized_notes, min_max_values = normalize_data(all_notes)
normalized_sequences = [(np.array(seq) - min_max_values[0]) / (np.maximum(min_max_values[1] - min_max_values[0], 1e-8)) for seq in sequences]

X, y = [], []
seq_length = 50
for seq in normalized_sequences:
    if len(seq) > seq_length:
        for i in range(len(seq) - seq_length):
            X.append(seq[i:i + seq_length])
            y.append(seq[i + seq_length])

X, y = np.array(X), np.array(y)

y_pitch = to_categorical(y[:, 0].astype(int), num_classes=128)
y_duration_velocity = y[:, 1:]

input_layer = Input(shape=(seq_length, 3))

lstm = LSTM(256, return_sequences=True)(input_layer)
lstm = Dropout(0.3)(lstm)
lstm = LSTM(256)(lstm)

pitch_output = Dense(128, activation="softmax", name="pitch_output")(lstm)

duration_velocity_output = Dense(2, activation="linear", name="duration_velocity_output")(lstm)

model = Model(inputs=input_layer, outputs=[pitch_output, duration_velocity_output])

model.compile(loss=["categorical_crossentropy", "mse"], optimizer="adam", metrics=["accuracy"])

model.fit(X, [y_pitch, y_duration_velocity], epochs=3, batch_size=64, validation_split=0.1)

model.save("music_gen_model.h5")
print("Model saved successfully!")
