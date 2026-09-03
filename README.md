# Machine Learning from Scratch

Dự án cá nhân lưu trữ quá trình tự học, nghiên cứu và tự cài đặt (implement from scratch) các thuật toán **Học máy (Machine Learning)** từ con số 0 bằng Python và NumPy, không sử dụng các thư viện black-box (như mô hình có sẵn của Scikit-Learn) cho phần cốt lõi của thuật toán.


## Mục tiêu dự án

- **Hiểu sâu bản chất toán học:** Nắm vững cách các thuật toán hoạt động bên dưới lớp vỏ bọc trừu tượng (Đại số tuyến tính, Giải tích vi phân, Tối ưu hóa, Xác suất thống kê).
- **Làm chủ kỹ thuật Vector hóa (Vectorization):** Sử dụng tối đa sức mạnh tính toán mảng nhiều chiều của `NumPy` thay cho các vòng lặp thủ công chậm chạp.
- **Tiền xử lý và trực quan hóa dữ liệu:** Rèn luyện kỹ năng làm sạch dữ liệu, chuẩn hóa đặc trưng (Feature Scaling) và trực quan hóa quá trình hội tụ/kết quả dự đoán bằng `Matplotlib` & `Pandas`.
- **Xây dựng nền tảng vững chắc:** Chuẩn bị tư duy logic vững vàng để tiến sâu hơn vào Deep Learning và các kiến trúc AI hiện đại.


## Cấu trúc thư mục

```text
Machine-Learning-from-scratch/
│
├── knn/
│   ├── data.csv                   # Tập dữ liệu chẩn đoán ung thư vú (Breast Cancer Wisconsin Diagnostic)
│   └── knn.ipynb                  # Cài đặt thuật toán K-Nearest Neighbors từ đầu
│
├── linear regression/
│   ├── Salary Data.csv            # Dữ liệu kinh nghiệm làm việc và mức lương tương ứng
│   └── linear_regression.ipynb    # Cài đặt Hồi quy tuyến tính đơn biến với Gradient Descent
│
└── README.md                      # Tài liệu hướng dẫn & lộ trình dự án
```


## Chi tiết các thuật toán đã triển khai

### 1. Hồi quy tuyến tính (Linear Regression)

- **Thư mục:** `linear regression`
- **File notebook:** `linear_regression.ipynb`
- **Tập dữ liệu:** `Salary Data.csv`
    - (30 mẫu: `YearsExperience` ➔ `Salary`)
- **Bài toán:** Dự đoán mức lương liên tục dựa trên số năm kinh nghiệm làm việc.
- **Cơ sở lý thuyết:**
  - **Hàm giả thuyết (Hypothesis function):**
    $$f_{w, b}(x) = w \cdot x + b$$
  - **Hàm mất mát (Mean Squared Error - MSE Cost Function):**
    $$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w, b}(x^{(i)}) - y^{(i)} \right)^2$$
  - **Đạo hàm riêng (Gradients):**
    $$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w, b}(x^{(i)}) - y^{(i)} \right) x^{(i)}$$
    $$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f_{w, b}(x^{(i)}) - y^{(i)} \right)$$
  - **Quy tắc cập nhật Gradient Descent:**
    $$w := w - \alpha \frac{\partial J}{\partial w}$$
    $$b := b - \alpha \frac{\partial J}{\partial b}$$
- **Kết quả thực nghiệm:**
  - Tốc độ học ($\alpha$): `0.001`, Số vòng lặp (epochs): `10000`.
  - Tham số học được tối ưu: $w \approx 9876.97, b \approx 22914.69$.
  - Đường hồi quy khớp chính xác xu hướng tăng lương tuyến tính theo năm kinh nghiệm.


### 2. K láng giềng gần nhất (K-Nearest Neighbors - KNN)

- **Thư mục:** `knn`
- **File notebook:** `knn.ipynb`
- **Tập dữ liệu:** `data.csv`
    - (569 mẫu, 30 đặc trưng y khoa về kích thước tế bào ung thư vú)
- **Bài toán:** Phân loại nhị phân khối u là **Lành tính (Benign - B)** hay **Ác tính (Malignant - M)**.
- **Cơ sở lý thuyết:**
  - **Khoảng cách Euclidean:**
    $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
  - **Cơ chế hoạt động:**
    1. Lưu trữ toàn bộ tập huấn luyện (Lazy Learning / Instance-based Learning).
    2. Với mỗi điểm dữ liệu mới, tính khoảng cách Euclidean tới mọi điểm trong tập huấn luyện.
    3. Chọn ra $k$ điểm có khoảng cách nhỏ nhất (`np.argsort`).
    4. Bầu cử đa số (Majority Voting) bằng `Counter` để đưa ra nhãn dự đoán có tần suất xuất hiện cao nhất.
- **Tiền xử lý dữ liệu:**
  - Loại bỏ các cột trống (`NaN`) và cột định danh (`id`).
  - Chuẩn hóa thang đo đặc trưng (`StandardScaler`) để tránh việc các đặc trưng có giá trị lớn lấn át khoảng cách Euclidean.
  - Phân chia tập huấn luyện và kiểm thử (80% Train / 20% Test, `random_state=2`).
- **Kết quả thực nghiệm:**
  - Độ chính xác (Accuracy) trên tập kiểm thử đạt **98.25%** với $k = 3$.
  - Trực quan hóa 2D phân bố điểm dữ liệu và nhãn dự đoán.


## Yêu cầu môi trường & Cài đặt

### 1. Yêu cầu
- **Python 3.8+**
- Các thư viện cần thiết:
  - `numpy`: Xử lý mảng và đại số tuyến tính
  - `pandas`: Đọc và thao tác với tập dữ liệu bảng
  - `matplotlib`: Trực quan hóa dữ liệu và biểu đồ học tập
  - `scikit-learn`: Sử dụng riêng cho các bước tiện ích tiền xử lý (`train_test_split`, `StandardScaler`)
  - `jupyter` / `notebook`: Môi trường tương tác dòng lệnh

### 2. Cài đặt các thư viện
Bạn có thể cài đặt toàn bộ các thư viện hỗ trợ bằng lệnh:

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

### 3. Khởi chạy dự án
Clone repository về máy và mở Jupyter Notebook:

```bash
git clone https://github.com/BaThanhHuynh/Machine-Learning-from-scratch.git
cd Machine-Learning-from-scratch
jupyter notebook
```

Sau đó duyệt vào từng thư mục (`knn/` hoặc `linear regression/`) và mở file `.ipynb` tương ứng để chạy từng ô lệnh (cell).


## Lộ trình phát triển (Roadmap)

Dự án đang liên tục được bổ sung các thuật toán mới theo lộ trình:

### Học có giám sát (Supervised Learning)
- **Linear Regression (Simple Gradient Descent)** - Đã hoàn thành
- **K-Nearest Neighbors (KNN Classifier)** - Đã hoàn thành
- **Multiple Linear Regression** (Hồi quy tuyến tính đa biến với Normal Equation & Vectorized Gradient Descent)
- **Logistic Regression** (Binary Classification, Sigmoid, Cross-Entropy Loss)
- **Softmax Regression** (Multinomial Classification)
- **Naive Bayes Classifier** (Gaussian Naive Bayes)
- **Decision Tree** (Information Gain, Entropy, Gini Impurity)
- **Random Forest** (Bagging & Feature Subsampling)
- **Support Vector Machine (SVM)** (Hinge Loss, Kernel trick cơ bản)

### Học không giám sát (Unsupervised Learning)
- **K-Means Clustering** (K-Means++ initialization, WCSS / Elbow method)
- **Principal Component Analysis (PCA)** (Eigenvectors, SVD, Dimensionality Reduction)

### Mạng nơ-ron cơ bản (Deep Learning Basics)
- **Perceptron** (Single-layer perceptron)
- **Multi-Layer Perceptron (MLP)** (Neural Network from scratch với Forward Pass & Backpropagation)


## Tài liệu tham khảo

1. **Machine Learning Specialization** – *Prof. Andrew Ng (Coursera / DeepLearning.AI)*
2. **Machine Learning Cơ Bản** – *Vũ Hữu Tiệp (machinelearningcoban.com)*
3. **Python Data Science Handbook** – *Jake VanderPlas*
4. **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** – *Aurélien Géron*


## Bản quyền (License)

Dự án được mở cho mục đích học tập và nghiên cứu cá nhân. Bạn hoàn toàn có thể tự do tham khảo, fork và phát triển thêm!