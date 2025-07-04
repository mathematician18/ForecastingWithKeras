# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import ipywidgets as widgets

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import shapiro, normaltest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import display


# -------------------------------
# 1. Data loading and cleaning
df = pd.read_csv('../data/hourly_electricity.csv', parse_dates=['DateTime'], index_col='DateTime')
df = df.sort_index().interpolate()  # simple interpolation of missing data


# -------------------------------
df.head(10)

# -------------------------------
# 2. Preprocessing
#  Demonstrate that Production is redundant

df['Production_calc'] = df[['Nuclear', 'Wind', 'Hydroelectric', 'Oil and Gas', 'Coal', 'Solar', 'Biomass']].sum(axis=1)
df[['Production', 'Production_calc']].head()

#  Check if they are the same
son_iguales = df['Production'].equals(df['Production_calc'])
print(f"¿Production y Production_calc son iguales? {son_iguales}")


# -------------------------------
#  Calculate the average relative difference (avoiding division by zero)
with np.errstate(divide='ignore', invalid='ignore'):
    diferencias_relativas = np.abs((df['Production'] - df['Production_calc']) / df['Production'])
    diferencias_relativas = np.nan_to_num(diferencias_relativas, nan=0.0)  # Manejar divisiones por cero
    mre = diferencias_relativas.mean()

print(f" Average relative difference (MRE): {mre:.6%}")


# -------------------------------
# 3. Feature selection excluding 'Production' because it is linearly dependent
features = ['Nuclear', 'Wind', 'Hydroelectric', 'Oil and Gas', 'Coal', 'Solar', 'Biomass', 'Consumption']

# -------------------------------
# 4. Preprocessing (data normalization)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), index=df.index, columns=features)


# -------------------------------
# 5. Creating sequences for LSTM modeling (Sequences from the last 24 hours to predict the next hour)
def create_sequences(data, target_col, seq_len=24):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data.iloc[i:i+seq_len].values)
        y.append(data.iloc[i+seq_len][target_col])
    return np.array(X), np.array(y)

SEQ_LEN = 24  
X, y = create_sequences(df_scaled, target_col='Consumption', seq_len=SEQ_LEN)


# -------------------------------
# 6. Division into Train and Test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------------
# 7. Definition of the LSTM model
# 64 units are chosen in the LSTM layer as a standard starting point that allows the model to capture complex relationships without being overly heavy.
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, len(features)), return_sequences=False),
    Dropout(0.2), #20% dropout to mitigate overfitting.
    Dense(32, activation='relu'), # ReLU prevents the problem of vanishing gradients and speeds up training
    Dense(1)
])
model.compile(optimizer='adam', loss='mse') # Adam is robust and efficient for complex problems

# -------------------------------
# 8. Training with Early Stopping to avoid overfitting
import time

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

start_time = time.time()  # Start of medition

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)
end_time = time.time()  # End of medition

elapsed_time = end_time - start_time
print(f"⏱️ Total training time: {elapsed_time:.2f} segundos")

# -------------------------------
# 9. Visualizing loss curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Models Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# -------------------------------
# 10. Model evaluation and forecasting
test_loss = model.evaluate(X_test, y_test)
print(f'Test MSE: {test_loss:.4f}')

y_pred = model.predict(X_test)

# -------------------------------
# Reverse rescaling to interpret results in original units
# a) Prediction already scaled, we reverse the scaling of 'Consumption' only
# Retrieve the scaler for Consumption
mean_c = scaler.mean_[features.index('Consumption')]
std_c = scaler.scale_[features.index('Consumption')]

# Reverse rescaling (compare in original units)
y_test_rescaled = y_test * std_c + mean_c
y_pred_rescaled = y_pred.flatten() * std_c + mean_c

# -------------------------------
# b) Calculation of metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

mean_consumption = y_test_rescaled.mean()
mae_percent = (mae / mean_consumption) * 100
rmse_percent = (rmse / mean_consumption) * 100

print(f"Absolute MAE: {mae:.2f} MW ({mae_percent:.2f}%)")
print(f"Absolute RMSE: {rmse:.2f} MW ({rmse_percent:.2f}%)")


# -------------------------------
# 11. Professional visualization of results
# Actual time index of the test set

test_index = df_scaled.index[-len(y_test_rescaled):]

plt.figure(figsize=(14,6))

# Graphic 1: Actual Consumption
plt.subplot(2,1,1)
plt.plot(test_index, y_test_rescaled, color='blue')
plt.title('Actual Consumption - Test Serie')
plt.ylabel('Consumption (MW)')
plt.xlabel('time')

# Graphic 2: Forecasting
plt.subplot(2,1,2)
plt.plot(test_index, y_pred_rescaled, color='orange')
plt.title('Consumption Forecasting - Test Serie')
plt.ylabel('Consumption (MW)')
plt.xlabel('Time')

plt.tight_layout()
plt.show()


# -------------------------------
# 12. Residual analysis
residuals = y_test_rescaled - y_pred_rescaled

# Residual temporal serie 
plt.figure(figsize=(12,4))
sns.lineplot(x=np.arange(len(residuals)), y=residuals, color='purple')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals over time')
plt.xlabel('Time (hrs)')
plt.ylabel('Error (MW)')
plt.show()

# Residual Histogram 
plt.figure(figsize=(6,4))
sns.histplot(residuals, kde=True, color='orange')
plt.title('Residuals Distribution')
plt.xlabel('Error (MW)')
plt.show()


# -------------------------------
# Análisis de Homoscedasticidad
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_pred_rescaled, y=residuals, alpha=0.5)
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals vs Prediction (Homoscedasticity)')
plt.xlabel('Forecasting (MW)')
plt.ylabel('Error (MW)')
plt.show()


# -------------------------------
# Statistical tests of residues
shapiro_p = shapiro(residuals)[1]
normaltest_p = normaltest(residuals)[1]
print(f"Test de Shapiro-Wilk p-value: {shapiro_p:.4f}")
print(f"Test de D'Agostino p-value: {normaltest_p:.4f}")


# -------------------------------
# 13. Iterative Forecasting
# Interactive forecasting function
def forecast_horizon(horizon_hours=1):
    """
    Produces a consumption forecast for the next 'horizon_hours'.
    Uses chained predictions based on the last known sequence.
    """
    last_sequence = df_scaled[-SEQ_LEN:].values.reshape(1, SEQ_LEN, len(features))
    predictions = []
    sequence = last_sequence.copy()
    
    for _ in range(horizon_hours):
        pred = model.predict(sequence)[0][0]
        predictions.append(pred)
        next_step = np.append(sequence[0, 1:, :], [[*sequence[0, -1, :-1], pred]], axis=0)
        sequence = next_step.reshape(1, SEQ_LEN, len(features))

    # Reverse rescaling of predictions
    mean_c = scaler.mean_[features.index('Consumption')]
    std_c = scaler.scale_[features.index('Consumption')]
    predictions_rescaled = np.array(predictions) * std_c + mean_c

    avg_pred = predictions_rescaled.mean()
    confidence = max(0, 100 - horizon_hours * 2)

    print(f"\n🔮 Average consumption forecast for the next few months {horizon_hours} horas: {avg_pred:.2f} MW")
    print(f"📉 Estimated confidence: {confidence:.1f}% (the longer the horizon, the lower the reliability)")

    plt.figure(figsize=(8,4))
    plt.plot(predictions_rescaled, marker='o', color='orange')
    plt.title(f'Consumption forecast - Next {horizon_hours} hours')
    plt.xlabel('Horizon (hrs)')
    plt.ylabel('Consumption (MW)')
    plt.grid()
    plt.show()

# Activación del widget interactivo
# widgets.interact(forecast_horizon, horizon_hours=widgets.IntSlider(value=1, min=1, max=24, step=1, description='Horizon (h)'));

if __name__ == "__main__":
    horizon_hours = int(input("Enter the forecast horizon (hours 1-24)): "))
    forecast_horizon(horizon_hours)


# -------------------------------
# 14.Mostrar resumen estructural del modelo
model.summary()

# Extraer pesos de cada capa
for i, layer in enumerate(model.layers):
    print(f"\n🔧 Capa {i} - {layer.name}")
    weights = layer.get_weights()
    for j, param in enumerate(weights):
        print(f"  Parámetro {j} - Forma: {param.shape}")


# -------------------------------


