# Bone Age Estimation from Hand Radiographs

An **end-to-end deep learning pipeline** for estimating bone age from left-hand X-rays by integrating object detection, medical image segmentation, and regression modeling.

---

## 📌 Overview
Bone age estimation is a crucial task in pediatric radiology for assessing growth and diagnosing endocrine or developmental disorders.  
This project implements a **hybrid deep learning and classical modeling pipeline** that:
- Detects regions of interest (ROIs) in X-rays
- Segments bone structures
- Extracts clinically relevant features
- Maps them to age predictions via regression models  

The approach outperforms many existing automated bone age estimation methods.

---

## ✨ Features
- **ROI Detection with YOLOv8**  
  Trained on 300 X-rays to detect 5 anatomical regions of interest.
- **Bone Segmentation with UNet++**  
  Custom-trained (100 annotated X-rays) for accurate bone masking.
- **Preprocessing**  
  - Bone orientation standardization (joint alignment + angle correction)  
  - ROI extraction using bounding boxes  
  - Edge refinement using Canny edge detection
- **Feature Engineering**  
  - Extracted bone width ratios (32 key points per ROI)  
  - Polynomial regression per ROI
- **Final Age Estimation**  
  Averaging outputs of ROI-specific models for robustness
- **Evaluation**  
  Achieved **MAE = 6.28 months** and **RMSE = 7.23 months** on 1800+ X-rays  
  (Normalized MAE = 0.0467, Normalized RMSE = 0.0538)

---

## 🛠 Tech Stack
- **Programming Language**: Python  
- **Deep Learning**: PyTorch, [segmentation-models-pytorch (SMP)](https://github.com/qubvel/segmentation_models.pytorch)  
- **Computer Vision**: OpenCV, YOLOv8, Canny edge detection  
- **Machine Learning**: scikit-learn, NumPy  
- **Other Tools**: Custom image preprocessing & feature extraction pipelines

---

## 📊 Results
| Metric | Value |
|--------|-------|
| MAE (months) | **6.28** |
| RMSE (months) | **7.23** |
| Normalized MAE | **0.0467** |
| Normalized RMSE | **0.0538** |

✅ Significantly outperforms existing automated bone age estimation methods.

---

## 📖 References & Acknowledgments
- Dataset: Left-hand radiographs (custom collected + annotated)  
- Models: YOLOv8 ([Ultralytics](https://github.com/ultralytics/ultralytics)), UNet++ ([Zhou et al.](https://arxiv.org/abs/1807.10165))  
- Libraries: PyTorch, SMP, OpenCV, scikit-learn  
- Inspired by classical radiological assessment techniques (Greulich & Pyle, Tanner-Whitehouse)

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
