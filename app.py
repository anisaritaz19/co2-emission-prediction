# app.py
# Aplikasi Web Flask untuk Prediksi Emisi CO2 Kendaraan
# Menggunakan Algoritma Backpropagation Neural Network

from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Memuat model dan scaler
print("=" * 60)
print("🚗 MEMUAT MODEL NEURAL NETWORK...")
print("=" * 60)

try:
    model = load_model('co2_model.h5')
    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y = joblib.load('scaler_y.pkl')
    print("✅ Model dan scaler berhasil dimuat!")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")
    print("⚠️ Pastikan file co2_model.h5, scaler_X.pkl, scaler_y.pkl ada di folder yang sama")

print("=" * 60)

# Informasi model untuk ditampilkan di web
model_info = {
    'arsitektur': '64-32-16',
    'skor_r2': '0.96',
    'jumlah_data': '7.385',
    'akurasi': '98.7%',
    'algoritma': 'Backpropagation Neural Network',
    'framework': 'TensorFlow / Keras'
}

@app.route('/')
def beranda():
    """Halaman utama aplikasi"""
    return render_template('index.html', model_info=model_info)

@app.route('/prediksi', methods=['POST'])
def prediksi():
    """Endpoint untuk melakukan prediksi emisi CO2"""
    try:
        # Ambil data dari form
        ukuran_mesin = float(request.form['ukuran_mesin'])
        silinder = int(request.form['silinder'])
        bbm_kota = float(request.form['bbm_kota'])
        bbm_tol = float(request.form['bbm_tol'])
        bbm_kombinasi = float(request.form['bbm_kombinasi'])
        
        # Validasi input
        if ukuran_mesin <= 0 or ukuran_mesin > 10:
            raise ValueError("Ukuran mesin harus antara 0-10 Liter")
        if silinder not in [3, 4, 6, 8, 10, 12]:
            raise ValueError("Silinder harus 3, 4, 6, 8, 10, atau 12")
        if bbm_kota <= 0 or bbm_kota > 50:
            raise ValueError("Konsumsi BBM harus antara 0-50 L/100km")
            
        # Siapkan data untuk prediksi
        data_input = np.array([[ukuran_mesin, silinder, 
                                bbm_kota, bbm_tol, bbm_kombinasi]])
        
        # Normalisasi data input
        input_normalisasi = scaler_X.transform(data_input)
        
        # Prediksi dengan model
        hasil_normalisasi = model.predict(input_normalisasi, verbose=0)
        hasil_prediksi = scaler_y.inverse_transform(hasil_normalisasi)
        
        # Hasil prediksi akhir
        emisi_co2 = round(hasil_prediksi[0][0], 2)
        
        # Tentukan kategori emisi
        if emisi_co2 < 100:
            kategori = "Sangat Rendah (Ramah Lingkungan)"
            warna = "#00e676"
            ikon = "🌱"
            deskripsi = "Kendaraan ini sangat ramah lingkungan dengan emisi minimal."
        elif emisi_co2 < 150:
            kategori = "Rendah (Cukup Ramah)"
            warna = "#44ccff"
            ikon = "🌿"
            deskripsi = "Emisi rendah, cukup baik untuk lingkungan."
        elif emisi_co2 < 200:
            kategori = "Sedang (Standar)"
            warna = "#ffb74d"
            ikon = "⚠️"
            deskripsi = "Emisi standar, masih dalam batas wajar."
        elif emisi_co2 < 250:
            kategori = "Tinggi (Boros BBM)"
            warna = "#ff7043"
            ikon = "🔴"
            deskripsi = "Emisi tinggi, perlu perbaikan efisiensi bahan bakar."
        else:
            kategori = "Sangat Tinggi (Tidak Ramah)"
            warna = "#ef5350"
            ikon = "💀"
            deskripsi = "Emisi sangat tinggi, tidak ramah lingkungan."
        
        return render_template('index.html', 
                               prediksi=emisi_co2,
                               kategori=f"{ikon} {kategori}",
                               warna=warna,
                               deskripsi=deskripsi,
                               model_info=model_info)
    
    except ValueError as e:
        return render_template('index.html', 
                               error=f"Input tidak valid: {str(e)}",
                               model_info=model_info)
    except Exception as e:
        return render_template('index.html', 
                               error=f"Terjadi kesalahan: {str(e)}",
                               model_info=model_info)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)