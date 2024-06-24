import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Veriyi yükleyin
df = pd.read_csv(r'../data/sukalitesi.csv')

# Verinin son halini kontrol edin
# print(df.head())
# print(df.info)

# Veriyi temizledik. 
df = df.dropna()  
print(df.info)

# Özellikler ve hedefler belirleyin
target = df['Salinity (ppt)']  # Salinity (ppt) özelliğini hedef değişken olarak belirleyin
features = df[['Salinity (ppt)', 'DissolvedOxygen (mg/L)', 'pH', 'SecchiDepth (m)', 'WaterDepth (m)', 'WaterTemp (C)', 'AirTemp (C)']]

# Veriyi ölçekleyin
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
scaled_target = scaler.fit_transform(np.array(target).reshape(-1, 1))

# Veriyi eğitim ve test setlerine ayırın
train_size = int(len(scaled_features) * 0.67)
test_size = len(scaled_features) - train_size
X_train, X_test = scaled_features[0:train_size], scaled_features[train_size:len(scaled_features)]
Y_train, Y_test = scaled_target[0:train_size], scaled_target[train_size:len(scaled_target)]

# Modeli oluşturun
model = Sequential()
model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1))

# Modeli derleyin
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitin
model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=1)

# Modeli değerlendirin
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Tahminleri geri ölçekleyin
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
Y_train_actual = scaler.inverse_transform(Y_train)
Y_test_actual = scaler.inverse_transform(Y_test)

# Sonuçları görselleştirin
plt.figure(figsize=(12, 6))
plt.plot(Y_train_actual, label='Gerçek Eğitim Değerleri')
plt.plot(train_predict, label='Eğitim Tahminleri')
plt.plot(range(len(Y_train_actual), len(Y_train_actual) + len(Y_test_actual)), Y_test_actual, label='Gerçek Test Değerleri')
plt.plot(range(len(Y_train_actual), len(Y_train_actual) + len(Y_test_actual)), test_predict, label='Test Tahminleri')
plt.xlabel('Zaman')
plt.ylabel('Değer')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))

# Eğitim verisi için gerçek ve tahmin edilen değerler
plt.subplot(2, 1, 1)
plt.plot(Y_train_actual, label='Gerçek Değerler')
plt.plot(train_predict, label='Tahminler')
plt.title('Eğitim Seti - Gerçek vs Tahmin Edilen Değerler')
plt.xlabel('Örnek Numarası')
plt.ylabel('Salinity (ppt)')
plt.legend()

# Test verisi için gerçek ve tahmin edilen değerler
plt.subplot(2, 1, 2)
plt.plot(Y_test_actual, label='Gerçek Değerler')
plt.plot(test_predict, label='Tahminler')
plt.title('Test Seti - Gerçek vs Tahmin Edilen Değerler')
plt.xlabel('Örnek Numarası')
plt.ylabel('Salinity (ppt)')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

# Eğitim seti için gerçek ve tahmin edilen değerlerin geri ölçeklendirilmiş hali
plt.plot(df.index[:len(Y_train_actual)], Y_train_actual, label='Gerçek Değerler (Eğitim)')
plt.plot(df.index[:len(train_predict)], train_predict, label='Tahminler (Eğitim)')

# Test seti için gerçek ve tahmin edilen değerlerin geri ölçeklendirilmiş hali
plt.plot(df.index[len(Y_train_actual):len(Y_train_actual) + len(Y_test_actual)], Y_test_actual, label='Gerçek Değerler (Test)')
plt.plot(df.index[len(train_predict):len(train_predict) + len(test_predict)], test_predict, label='Tahminler (Test)')

plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.xlabel('Tarih')
plt.ylabel('Salinity (ppt)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"Original scaled features shape: {scaled_features.shape}")
print(f"Train predictions shape: {train_predict.shape}")
print(f"Test predictions shape: {test_predict.shape}")
print(f"Train predict plot range: {look_back}:{len(train_predict) + look_back}")
print(f"Test predict plot range: {len(train_predict) + (look_back * 2) + 1}:{len(scaled_features) - 1}")

