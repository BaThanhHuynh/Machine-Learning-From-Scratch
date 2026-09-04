# Machine Learning From Scratch

Dự án tự học và cài đặt các thuật toán Machine Learning từ con số 0 bằng Python và NumPy, không sử dụng các mô hình đóng gói sẵn của Scikit-Learn nhằm hiểu sâu bản chất toán học và thuật toán tối ưu.


## Mục tiêu dự án

- **Bản chất toán học:** Nắm vững giải tích vi phân, đại số tuyến tính, hàm mất mát và thuật toán tối ưu hóa.
- **Kỹ thuật vector hóa:** Tối ưu hóa tính toán trên ma trận/vector bằng NumPy, hạn chế tối đa vòng lặp.
- **Xử lý dữ liệu:** Tự xây dựng các bước tiền xử lý (chuẩn hóa đặc trưng, xử lý dữ liệu thiếu) và trực quan hóa kết quả bằng Matplotlib.


## Cấu trúc thư mục

```text
Machine-Learning-From-Scratch/
├── knn/
│   ├── data.csv                   # Tập dữ liệu ung thư vú (Breast Cancer Diagnostic)
│   └── knn.ipynb                  # Thuật toán K-Nearest Neighbors
├── linear regression/
│   ├── Salary Data.csv            # Dữ liệu số năm kinh nghiệm và mức lương
│   └── linear_regression.ipynb    # Hồi quy tuyến tính đơn biến với Gradient Descent
├── logistic regression/
│   ├── data_synthetic.csv         # Dữ liệu điểm thi và kết quả tuyển sinh
│   └── logistic_regression.ipynb  # Hồi quy Logistic với Binary Cross-Entropy
└── README.md
``` 


## Các thuật toán đã triển khai

### 1. Hồi quy tuyến tính (Linear Regression)

- **Mã nguồn:** `linear regression/linear_regression.ipynb`
- **Dữ liệu:** `Salary Data.csv` (30 mẫu, dự đoán `Salary` theo `YearsExperience`).
- **Toán học:**
  - Giả thuyết: $f_{w,b}(x) = w x + b$
  - Hàm mất mát MSE: $J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})^2$
  - Đạo hàm:
    $$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) x^{(i)}, \quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$$
  - Cập nhật Gradient Descent: $w := w - \alpha \frac{\partial J}{\partial w}, \quad b := b - \alpha \frac{\partial J}{\partial b}$
- **Kết quả:** Learning rate $\alpha = 0.001$, 10.000 epochs. Nghiệm hội tụ: $w \approx 9876.97, b \approx 22914.69$.


### 2. Hồi quy Logistic (Logistic Regression)

- **Mã nguồn:** `logistic regression/logistic_regression.ipynb`
- **Dữ liệu:** `data_synthetic.csv` (200 mẫu, phân loại trúng tuyển `Admitted` dựa trên `Math_Score` và `English_Score`).
- **Toán học:**
  - Hàm Sigmoid: $g(z) = \frac{1}{1 + e^{-z}}$ (sử dụng `np.clip(z, -250, 250)` để chống tràn số).
  - Giả thuyết: $\hat{y} = g(X W)$ với $X$ đã bổ sung cột bias $x_0 = 1$.
  - Hàm mất mát Binary Cross-Entropy:
    $$J(W) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)} + \epsilon) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)} + \epsilon) \right]$$
  - Đạo hàm ma trận: $\nabla_W J = \frac{1}{m} X^T (\hat{y} - y)$
  - Ranh giới quyết định (Decision Boundary): $w_0 + w_1 x_1 + w_2 x_2 = 0 \implies x_2 = -\frac{w_0 + w_1 x_1}{w_2}$
- **Tiền xử lý:** Chuẩn hóa Z-score trực tiếp bằng NumPy ($X_{\text{scaled}} = \frac{X - \mu}{\sigma}$).
- **Kết quả:** Learning rate $\alpha = 0.01$, 10.000 epochs. Chi phí giảm từ `0.6899` về `0.0595`. Độ chính xác đạt **98.00%**.


### 3. K láng giềng gần nhất (K-Nearest Neighbors)

- **Mã nguồn:** `knn/knn.ipynb`
- **Dữ liệu:** `data.csv` (569 mẫu, 30 đặc trưng y tế, phân loại khối u lành tính B / ác tính M).
- **Toán học & Giải thuật:**
  - Khoảng cách Euclidean: $d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$
  - Cơ chế Lazy Learning: Tính khoảng cách từ điểm mới tới toàn bộ tập huấn luyện, lấy $k$ điểm gần nhất và bỏ phiếu đa số (Majority Voting).
- **Tiền xử lý:** Chuẩn hóa đặc trưng bằng `StandardScaler`, chia tập Train/Test tỉ lệ 80/20.
- **Kết quả:** Độ chính xác đạt **98.25%** với $k = 3$.


## Cài đặt & Sử dụng 

### 1. Yêu cầu môi trường
- Python 3.8+
- Các thư viện cần thiết:
```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

### 2. Chạy dự án
```bash
git clone https://github.com/BaThanhHuynh/Machine-Learning-from-scratch.git
cd Machine-Learning-from-scratch
jupyter notebook
```
Mở notebook tương ứng trong từng thư mục để xem mã nguồn và kết quả chạy trực quan.


## Lộ trình phát triển (Roadmap)

### Học có giám sát (Supervised Learning)
- [x] Linear Regression (Gradient Descent)
- [x] Logistic Regression (Binary Cross-Entropy & Decision Boundary)
- [x] K-Nearest Neighbors (KNN Classifier)
- [ ] Multiple Linear Regression (Vectorized Gradient Descent & Normal Equation)
- [ ] Softmax Regression (Multinomial Classification)
- [ ] Naive Bayes Classifier
- [ ] Decision Tree
- [ ] Random Forest
- [ ] Support Vector Machine (SVM)

### Học không giám sát (Unsupervised Learning)
- [ ] K-Means Clustering
- [ ] Principal Component Analysis (PCA)

### Mạng nơ-ron cơ bản (Deep Learning Basics)
- [ ] Perceptron
- [ ] Multi-Layer Perceptron (MLP with Backpropagation)


## Tài liệu tham khảo

1. Machine Learning Specialization – Andrew Ng (Coursera / DeepLearning.AI)
2. Machine Learning Cơ Bản – Vũ Hữu Tiệp (machinelearningcoban.com)
3. Python Data Science Handbook – Jake VanderPlas


## Giấy phép (License)

Dự án được phân phối cho mục đích học tập và nghiên cứu cá nhân.
