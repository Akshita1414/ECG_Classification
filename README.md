# 💡 GNN-Based Feature Selection for ECG Classification

This project focuses on the application of Graph Neural Networks (GNNs), particularly Graph Attention Networks (GAT), for interpretable feature selection in ECG signal classification using the MIT-BIH Arrhythmia dataset. Our method extracts ECG features, models their relationships as a graph, ranks them using attention scores, and evaluates downstream classification using machine learning models.

---

## 📚 Dataset

* MIT-BIH Arrhythmia Database (PhysioNet)
* 48 annotated ECG recordings, 2 channels per recording
* Sampling rate: 360 Hz
* Each beat is annotated and categorized (e.g., N: Normal, V: Ventricular, S: Supraventricular, F: Fusion, Q: Unknown)

---

## ⚙️ Preprocessing

* Baseline wander removal (high-pass filtering)
* Powerline interference suppression (notch filter at 50/60 Hz)
* High-frequency noise reduction (low-pass filtering)
* R-peak detection (Pan-Tompkins or bioSPPy)
* Normalization and beat segmentation

---

## 🧠 Feature Extraction

Features extracted include:

* Morphological: QRS duration, P-wave, T-wave amplitude, ST-segment
* HRV-based: RR intervals, SDNN, RMSSD, PNN50, LF/HF ratio
* Time-domain: R-peak amplitude, beat duration
* Frequency-domain: Wavelet coefficients (Daubechies), FFT bands
* Others: Signal entropy, area under curve

---

## 🔗 Graph Construction

* Nodes = ECG features
* Edges = Feature correlations (Pearson / Mutual Information) or domain-defined
* Fully connected graph or thresholded edges based on similarity
<img width="304" height="233" alt="image" src="https://github.com/user-attachments/assets/4f78861a-2bf5-4ca5-9cce-b132ec989064" />

---

## 🧠 GNN Pipeline

* Graph Attention Network (GAT) used for supervised classification
* Attention scores used to rank features (top-k selected)
* Feature selection is interpretable and explainable
* No need for external explainers (optional SHAP/GNNExplainer used)

---

## 🤖 Classifiers & Evaluation

Selected features used to train:

* SVM (Support Vector Machine)
* Random Forest
* XGBoost
* MLP (Multilayer Perceptron)
* LightGBM

Compared against:

* LASSO
* Ridge Regression
  
 <img width="316" height="271" alt="image" src="https://github.com/user-attachments/assets/a4e80245-4f35-459d-a1d4-76ed3f24d202" />


Metrics:

* Accuracy, F1-score, Precision, Recall
* Feature subset size
* Time efficiency

---

## 📊 Results Summary

* GEFA-inspired method outperformed LASSO and Ridge in accuracy and interpretability.
* Reduced feature set size with minimal performance loss.
* Enhanced explainability through attention visualization.
  <img width="1080" height="606" alt="image" src="https://github.com/user-attachments/assets/4b002fe2-3db3-4ebf-8c35-7053be7e3c3c" />

---

## 🧪 Project Team

This project is part of a larger collaborative research effort involving multiple ECG-related tasks such as:

* Atrial Fibrillation Detection
* Sleep Apnea Detection
* Emotion/Stress Detection
* Real-Time ECG Monitoring via Edge AI
* Gender/Age Prediction from ECG Morphology

Our team focused on the Feature Selection for ECG Classification module using GNNs.

Team Members:

* Akshita Chauhan – GNN modeling, classifier evaluation,result benchmarking,& research draft
* Yatish Garg – Feature extraction, preprocessing,& Graph construction, Documentation,& visualization
—

## 📄 Research Paper (In Progress)

“Comparison of Feature Selection Methods for ECG Classification using GNNs”

* Includes comparative analysis with LASSO and Ridge
* Focuses on explainable AI and biomedical feature ranking

—

## 🧰 Tools & Libraries

* Python, NumPy, Pandas, SciPy, Scikit-learn
* PyTorch, PyTorch Geometric (GAT)
* Matplotlib, Seaborn
* MIT-BIH ECG Toolkit, BioSPPy, WFDB


