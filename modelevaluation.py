from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Assuming you have your test data prepared as X_test and y_test

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Visualize the predictions
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Ground Truth')
plt.plot(y_pred, label='Predictions')
plt.title('Ground Truth vs Predictions')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()
