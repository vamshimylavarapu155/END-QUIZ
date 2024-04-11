from keras.models import Sequential
from keras.layers import LSTM, Dropout

# Define the model
model = Sequential()

# Add LSTM layers
model.add(LSTM(units=128, return_sequences=True, input_shape=(timesteps, input_dim)))  # First LSTM layer
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting

model.add(LSTM(units=64, return_sequences=True))  # Second LSTM layer
model.add(Dropout(0.2))  # Dropout layer

model.add(LSTM(units=32))  # Third LSTM layer
model.add(Dropout(0.2))  # Dropout layer

# Output layer
model.add(Dense(units=output_dim, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()
