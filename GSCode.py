# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt

# 1. Carregar o arquivo e considerar apenas o tipo_consumo residencial
# Lendo o arquivo CSV
file_path = '/mnt/data/br_mme_consumo_energia_eletrica_uf.csv.gz'
data = pd.read_csv(file_path, compression='gzip')

# Filtrar apenas consumo residencial
data_residencial = data[data['tipo_consumo'] == 'Residencial']

# 2. Agregar por estado (consumo total e número de consumidores)
aggregated_data = data_residencial.groupby(['uf', 'ano_mes']).agg(
    consumo_total=('consumo_kwh', 'sum'),
    num_consumidores=('consumidores', 'sum')
).reset_index()

# Ordenar por data
aggregated_data['ano_mes'] = pd.to_datetime(aggregated_data['ano_mes'], format='%Y-%m')
aggregated_data = aggregated_data.sort_values('ano_mes')

# 3. Pré-processamento dos dados
# Foco no consumo total para análise de séries temporais
series_data = aggregated_data.groupby('ano_mes')['consumo_total'].sum().values

# Normalizar os dados no intervalo [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
series_data_scaled = scaler.fit_transform(series_data.reshape(-1, 1))

# Construir sequências temporais (12 meses anteriores para prever o próximo valor)
def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(series_data_scaled)

# Divisão em conjunto de treino e teste (80% treino, 20% teste)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Redimensionar os dados para formato LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 4. Construir o modelo LSTM
model = Sequential([
    LSTM(50, activation='relu', return_sequences=False, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    Dense(1, activation='linear')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Treinamento do modelo
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=1
)

# Avaliação do modelo no conjunto de teste
y_pred = model.predict(X_test)
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = scaler.inverse_transform(y_pred)

# Cálculo do MSE e RMSE
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Gráfico comparando valores reais e preditos
plt.figure(figsize=(14, 7))
plt.plot(y_test_original, label="Valores Reais", color="blue")
plt.plot(y_pred_original, label="Previsões", color="red", linestyle="--")
plt.title("Comparação de Valores Reais e Previstos")
plt.xlabel("Período")
plt.ylabel("Consumo Total (kWh)")
plt.legend()
plt.show()

# 6. Previsão para os próximos 12 meses
last_sequence = series_data_scaled[-12:].reshape(1, -1, 1)
future_predictions = []
for _ in range(12):
    pred = model.predict(last_sequence)
    future_predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[:, 1:, :], [[pred]], axis=1)

# Reverter normalização das previsões
future_predictions_original = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Gráfico das previsões futuras
plt.figure(figsize=(14, 7))
plt.plot(range(len(series_data_scaled)), scaler.inverse_transform(series_data_scaled), label="Histórico")
plt.plot(range(len(series_data_scaled), len(series_data_scaled) + 12), future_predictions_original, label="Previsões Futuras", color="green")
plt.title("Previsão de Consumo para os Próximos 12 Meses")
plt.xlabel("Período")
plt.ylabel("Consumo Total (kWh)")
plt.legend()
plt.show()
