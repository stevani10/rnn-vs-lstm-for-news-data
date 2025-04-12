# Pemodelan Sekuens dengan RNN dan LSTM untuk Klasifikasi Teks

Proyek ini mengeksplorasi dan membandingkan performa dua jenis model neural network — **RNN (Recurrent Neural Network)** dan **LSTM (Long Short-Term Memory)** — untuk tugas klasifikasi teks menggunakan dataset **AG News**. Penelitian ini bertujuan untuk menganalisis efek dari dimensi word embedding dan panjang hidden state terhadap akurasi, serta memvisualisasikan representasi hidden state dari kedua model.


## Dataset

Dataset yang digunakan adalah [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset), berisi artikel berita berbahasa Inggris dari lebih dari 2.000 sumber di seluruh dunia:

- **Total**: 127.600 artikel (120.000 train, 7.600 test)
- **Label Kelas**:
  - `0` - World
  - `1` - Sports
  - `2` - Business
  - `3` - Sci/Tech

> Data diunduh dari:
> - [train.csv](https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv)  
> - [test.csv](https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv)

---

## Implementasi

- Bahasa: Python
- Framework: PyTorch
- Word Embedding: [GloVe 6B](https://nlp.stanford.edu/projects/glove/) (100 dimensi)
- Dimensi Hidden State: 128, 256, 512
- Evaluasi: Akurasi, Precision, Recall, F1 Score (macro average)

---

## Arsitektur & Eksperimen

### 1. **Modeling**
- Dua model dibangun: `RNNClassifier` dan `LSTMClassifier`
- Embedding layer menggunakan pre-trained GloVe
- Output layer → Softmax ke 4 kelas AG News

### 2. **Evaluasi Metrik**

| Hidden Dim | Model | Precision | Recall | F1-Score |
|------------|--------|-----------|--------|----------|
| 128        | RNN    | 0.3733    | 0.2390 | 0.1660   |
| 128        | LSTM   | 0.9213    | 0.9213 | 0.9210   |
| 256        | RNN    | 0.2297    | 0.2889 | 0.2161   |
| 256        | LSTM   | 0.9137    | 0.9129 | 0.9130   |
| 512        | RNN    | 0.1608    | 0.2649 | 0.1539   |
| 512        | LSTM   | 0.9148    | 0.9150 | 0.9148   |

---

## 📌 Insight Utama

- LSTM secara konsisten mengungguli RNN di semua hidden dimension.
- Semakin besar dimensi hidden, semakin jelas keunggulan LSTM terlihat secara **visual** dan **numerik**.
- RNN cenderung tidak stabil, terutama di hidden dimensi besar, karena tidak memiliki mekanisme gate seperti LSTM.


