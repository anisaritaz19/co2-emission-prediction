# train_model.py
# Program untuk melatih model Backpropagation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚗 PREDIKSI EMISI CO2 KENDARAAN DENGAN BACKPROPAGATION")
print("=" * 60)

# 1. LOAD DATASET
print("\n📂 Langkah 1: Memuat dataset...")
df = pd.read_csv('CO2 Emissions_Canada.csv')
print(f"✅ Dataset berhasil dimuat!")
print(f"📊 Jumlah data: {len(df)} baris")
print(f"📊 Jumlah kolom: {len(df.columns)} kolom")
print(f"\n📋 5 data pertama:")
print(df.head())

# 2. EDA (Exploratory Data Analysis)
print("\n📊 Langkah 2: Eksplorasi Data...")
print(f"\n🔍 Statistik deskriptif:")
print(df.describe())

print(f"\n🔍 Cek missing values:")
print(df.isnull().sum())

# 3. PILIH FITUR
print("\n🎯 Langkah 3: Memilih fitur untuk model...")

feature_columns = [
    'Engine Size(L)',
    'Cylinders',
    'Fuel Consumption City (L/100 km)',
    'Fuel Consumption Hwy (L/100 km)',
    'Fuel Consumption Comb (L/100 km)'
]

# Cek apakah kolom ada
for col in feature_columns:
    if col in df.columns:
        print(f"   ✅ {col} - OK")
    else:
        print(f"   ❌ {col} - TIDAK DITEMUKAN")

X = df[feature_columns].values
y = df['CO2 Emissions(g/km)'].values.reshape(-1, 1)

print(f"\n✅ Fitur (X) shape: {X.shape}")
print(f"✅ Target (y) shape: {y.shape}")

# 4. NORMALISASI DATA
print("\n🔄 Langkah 4: Normalisasi data...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

print("✅ Data berhasil dinormalisasi")

# 5. SPLIT DATA TRAINING & TESTING
print("\n📊 Langkah 5: Split data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"✅ Data training: {X_train.shape[0]} samples")
print(f"✅ Data testing: {X_test.shape[0]} samples")

# 6. MEMBANGUN MODEL BACKPROPAGATION
print("\n🧠 Langkah 6: Membangun model Backpropagation...")

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)

print("✅ Model berhasil dibangun")
print(f"\n📋 Arsitektur Model:")
model.summary()

# 7. TRAINING MODEL
print("\n🏋️ Langkah 7: Melatih model...")
print("=" * 60)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# 8. EVALUASI MODEL
print("\n" + "=" * 60)
print("📊 Langkah 8: Evaluasi model")
print("=" * 60)

test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"📉 Loss (MSE) pada data uji: {test_loss:.4f}")
print(f"📉 MAE pada data uji: {test_mae:.4f}")

rmse = np.sqrt(test_loss)
print(f"📉 RMSE pada data uji: {rmse:.4f}")

# Prediksi
y_pred_scaled = model.predict(X_test)
y_test_actual = scaler_y.inverse_transform(y_test)
y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)

r2 = r2_score(y_test_actual, y_pred_actual)
print(f"📈 R² Score: {r2:.4f}")

# 9. VISUALISASI
print("\n📈 Langkah 9: Membuat visualisasi...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Grafik Loss
axes[0].plot(history.history['loss'], label='Training Loss', color='blue')
axes[0].plot(history.history['val_loss'], label='Validation Loss', color='red')
axes[0].set_title('Model Loss (MSE)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# Grafik MAE
axes[1].plot(history.history['mae'], label='Training MAE', color='blue')
axes[1].plot(history.history['val_mae'], label='Validation MAE', color='red')
axes[1].set_title('Model MAE')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].legend()
axes[1].grid(True)

# Scatter plot Prediksi vs Aktual
axes[2].scatter(y_test_actual, y_pred_actual, alpha=0.5, color='green')
axes[2].plot([y_test_actual.min(), y_test_actual.max()], 
             [y_test_actual.min(), y_test_actual.max()], 
             'r--', lw=2)
axes[2].set_title('Prediksi vs Aktual')
axes[2].set_xlabel('Nilai Aktual (g/km)')
axes[2].set_ylabel('Nilai Prediksi (g/km)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
plt.show()
print("✅ Grafik disimpan sebagai 'training_results.png'")

# 10. SIMPAN MODEL
print("\n💾 Langkah 10: Menyimpan model...")
model.save('co2_model.h5')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ Model disimpan sebagai 'co2_model.h5'")
print("✅ Scaler disimpan sebagai 'scaler_X.pkl' dan 'scaler_y.pkl'")

# 11. CONTOH PREDIKSI
print("\n🔮 Langkah 11: Contoh prediksi data baru...")

# Data input: [Engine Size, Cylinders, Fuel City, Fuel Hwy, Fuel Comb]
new_data = np.array([[2.0, 4, 9.9, 6.7, 8.5]])

new_data_scaled = scaler_X.transform(new_data)
predicted_scaled = model.predict(new_data_scaled)
predicted_actual = scaler_y.inverse_transform(predicted_scaled)

print(f"\n📝 Data kendaraan:")
print(f"   - Ukuran mesin: 2.0 Liter")
print(f"   - Jumlah silinder: 4")
print(f"   - Konsumsi BBM kota: 9.9 L/100km")
print(f"   - Konsumsi BBM tol: 6.7 L/100km")
print(f"   - Konsumsi BBM kombinasi: 8.5 L/100km")
print(f"\n🎯 Prediksi emisi CO2: {predicted_actual[0][0]:.2f} g/km")

print("\n" + "=" * 60)
print("✅ PROGRAM TRAINING SELESAI!")
print("=" * 60)