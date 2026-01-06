# 🍄 Mushroom Classification Dashboard  
**Final Project – Data Mining and Visualization**

## 📌 Project Overview
Proyek ini bertujuan untuk melakukan **klasifikasi data jamur (mushroom)** menggunakan metode **Machine Learning Random Forest**, serta menyajikan hasil analisis dalam bentuk **dashboard interaktif**.  
Dashboard dikembangkan untuk memvisualisasikan karakteristik data dan hasil klasifikasi secara informatif dan mudah dipahami.

Dataset mushroom diolah melalui beberapa tahapan data mining, mulai dari preprocessing, pemodelan klasifikasi, hingga visualisasi hasil.

---

## 🎯 Objectives
- Membangun model klasifikasi jamur menggunakan **Random Forest**
- Mengevaluasi performa model klasifikasi
- Menyajikan hasil analisis dalam bentuk **dashboard visual**
- Mengintegrasikan analisis data (Python) dan visualisasi (R)

---

## 📂 Project Structure
```
├── data/
│ └── mushroom_dataset.csv # Dataset mushroom
│
├── python/
│ └── mushroom_rf_analysis.ipynb # Analisis & klasifikasi (Python)
│
├── r-dashboard/
│ └── app.R # Dashboard Shiny (R)
│
└── README.md
```

## 🧠 Methodology
### 1. Data Preprocessing
- Penanganan data kategorik
- Encoding variabel
- Pembagian data latih dan data uji

### 2. Classification Model
- Algoritma: **Random Forest**
- Bahasa: **Python**
- Evaluasi model menggunakan metrik seperti:
  - Accuracy
  - Confusion Matrix
  - (Opsional: Precision, Recall, F1-score)

### 3. Data Visualization
- Dashboard interaktif menggunakan **R Shiny**
- Visualisasi:
  - Distribusi fitur jamur
  - Hasil klasifikasi
  - Ringkasan performa model

---

## 🛠️ Tools & Technologies
- **Python**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib / seaborn
- **R**
  - shiny
  - ggplot2
  - dplyr
- **Machine Learning**
  - Random Forest Classifier

---

## 📊 Output
- Model klasifikasi Random Forest untuk data mushroom
- Dashboard interaktif yang menampilkan:
  - Eksplorasi data
  - Hasil klasifikasi jamur
  - Insight dari model

---

## 🚀 How to Run
### Python Analysis
1. Buka file `.ipynb` di folder `python/`
2. Jalankan seluruh cell untuk melakukan preprocessing dan klasifikasi

### R Dashboard
1. Buka folder `r-dashboard`
2. Jalankan file `app.R`
3. Dashboard akan tampil di browser

---

## 👩‍💻 Author
**Shafa Ashari**  
Final Project – Data Mining and Visualization

---

## 📎 Notes
Proyek ini dibuat untuk keperluan akademik dan pembelajaran dalam penerapan data mining, machine learning, dan visualisasi data.
