# � Big Data E-commerce Analytics Platform

> **Hệ thống phân tích dữ liệu mua sắm trực tuyến với khuyến nghị sản phẩm thông minh**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green)](https://flask.palletsprojects.com)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-orange)](https://spark.apache.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green)](https://mongodb.com)
[![HUIT](https://img.shields.io/badge/University-HUIT-red)](https://huit.edu.vn)

## 🎯 Mô tả Dự án
Dự án **Big Data Analytics** phân tích dữ liệu thương mại điện tử để hiểu hành vi khách hàng, dự đoán xu hướng và xây dựng hệ thống đề xuất sản phẩm thông minh. Được phát triển tại **Đại học Công Thương TP.HCM (HUIT)** cho môn học Big Data Analytics.

### 🏆 Mục tiêu
- Phân tích hành vi mua sắm của khách hàng
- Xây dựng hệ thống đề xuất sản phẩm thông minh
- Phát hiện xu hướng và mô hình tiêu dùng
- Tạo dashboard trực quan cho doanh nghiệp
- Demo web tương tác cho người dùng cuối

### 🔧 Công nghệ Sử dụng
- **Big Data**: Apache Spark, Hadoop, Kafka
- **Machine Learning**: Spark MLlib, Collaborative Filtering, Content-based Filtering
- **Database**: MongoDB, Redis (caching)
- **Web Framework**: Flask/FastAPI (Backend), React (Frontend)
- **Visualization**: Plotly, D3.js, Apache Superset
- **Containerization**: Docker, Docker Compose
- **Cloud**: AWS/Azure (tùy chọn)

## 📁 Cấu trúc Dự án

```
bigdata_project/
├── data/                          # Dữ liệu thô và đã xử lý
│   ├── raw/                      # Dữ liệu gốc
│   ├── processed/                # Dữ liệu đã xử lý
│   └── sample/                   # Dữ liệu mẫu
├── src/                          # Mã nguồn chính
│   ├── data_processing/          # Xử lý dữ liệu
│   ├── ml_models/               # Mô hình máy học
│   ├── recommendation_engine/    # Engine đề xuất
│   └── analytics/               # Phân tích dữ liệu
├── web_demo/                     # Ứng dụng web demo
│   ├── backend/                 # API Backend
│   ├── frontend/                # Giao diện người dùng
│   └── static/                  # Tài nguyên tĩnh
├── notebooks/                    # Jupyter notebooks
├── config/                       # Cấu hình
├── docker/                       # Docker configurations
├── docs/                        # Tài liệu
└── scripts/                     # Scripts tiện ích
```

## 🚀 Bắt đầu Nhanh

### 1. Cài đặt Dependencies
```bash
pip install -r requirements.txt
```

### 2. Khởi chạy với Docker
```bash
docker-compose up -d
```

### 3. Chạy Data Pipeline
```bash
python scripts/run_pipeline.py
```

### 4. Khởi động Web Demo
```bash
cd web_demo/backend
python app.py
```

## � Big Data E-commerce Analytics Platform

> **Hệ thống phân tích dữ liệu mua sắm trực tuyến với khuyến nghị sản phẩm thông minh**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green)](https://flask.palletsprojects.com)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-orange)](https://spark.apache.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-green)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 **Tổng quan**

Đây là một hệ thống **Big Data Analytics** hoàn chỉnh để phân tích dữ liệu thương mại điện tử, được xây dựng với:
- **Apache Spark** để xử lý dữ liệu lớn
- **Machine Learning** cho hệ thống khuyến nghị
- **Web Demo** với giao diện thân thiện
- **Vietnamese Support** hoàn chỉnh

## ✨ **Tính năng chính**

### 1. Thu thập & Xử lý Dữ liệu
- Import dữ liệu từ CSV/JSON
- Làm sạch và chuẩn hóa dữ liệu
- Xử lý streaming data với Kafka

### 2. Phân tích Hành vi Khách hàng
- Phân khúc khách hàng (Customer Segmentation)
- Phân tích giỏ hàng (Market Basket Analysis)
- Phân tích chu kỳ mua sắm

### 3. Hệ thống Đề xuất
- Collaborative Filtering
- Content-based Filtering
- Hybrid Recommendation System
- Real-time recommendations

### 4. Dashboard & Báo cáo
- Biểu đồ xu hướng bán hàng
- Phân tích sản phẩm hot
- Báo cáo doanh thu
- Heatmap hành vi người dùng

### 5. Web Demo
- Giao diện tìm kiếm sản phẩm
- Hệ thống đề xuất cá nhân
- Dashboard quản trị
- API RESTful

## 📈 Kết quả Mong đợi
- Độ chính xác đề xuất: >85%
- Tăng tỷ lệ click-through: 15-20%
- Cải thiện trải nghiệm người dùng
- Báo cáo phân tích chi tiết

## 🎓 Ứng dụng Học tập
Dự án này được thiết kế cho môn Big Data, minh họa:
- Xử lý dữ liệu lớn với Spark
- Áp dụng Machine Learning trong thực tế
- Xây dựng hệ thống phân tán
- Phát triển ứng dụng web tích hợp

## 📝 Tài liệu
- [Hướng dẫn Cài đặt](docs/installation.md)
- [Tài liệu API](docs/api.md)
- [Hướng dẫn Deployment](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## 👥 Tác giả
- **Tên sinh viên**: Lê Văn Thành
- **MSSV**: 2001224717  
- **Trường**: Đại học Công Thương TP.HCM (HUIT - University of Industry and Trade Ho Chi Minh City)
- **Khoa**: Công nghệ Thông tin
- **Môn học**: Big Data Analytics - Học kỳ 7
- **Năm học**: 2024-2025

## 📄� License
MIT License - Dự án giáo dục