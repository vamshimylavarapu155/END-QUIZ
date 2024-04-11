from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define a function to create the LSTM model
def create_model(learning_rate=0.001, lstm_units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units=lstm_units, input_shape=(n_steps, n_features)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Create a KerasRegressor based on the defined model
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the hyperparameters to search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'lstm_units': [50, 100, 150],
    'dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100, 150]
}

# Perform grid search
grid_search = GridSearchCV(est
