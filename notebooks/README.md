# 🎓 HUIT Big Data Project - E-commerce Analytics

## 📋 Hướng dẫn sử dụng Project

### 🚀 Quick Start

1. **Cài đặt môi trường:**
   ```bash
   cd bigdata_project
   pip install -r requirements.txt
   ```

2. **Chạy pipeline hoàn chỉnh:**
   ```bash
   python scripts/run_pipeline.py
   ```

3. **Truy cập web demo:**
   - Mở browser: http://localhost:5000
   - Xem analytics dashboard và recommendation system

### 📊 Jupyter Notebooks

#### 1. `ecommerce_analysis.ipynb` - Comprehensive Data Analysis
- **Mục đích:** Phân tích toàn diện dữ liệu e-commerce
- **Nội dung:**
  - Exploratory Data Analysis (EDA)
  - Customer Segmentation (RFM Analysis)  
  - Product Performance Analysis
  - Market Basket Analysis
  - Business Intelligence Insights

**Cách sử dụng:**
```bash
# Khởi động Jupyter
jupyter notebook notebooks/ecommerce_analysis.ipynb

# Hoặc sử dụng VS Code
# Mở file .ipynb trong VS Code và chạy từng cell
```

**Highlights:**
- 📊 Interactive visualizations với Plotly
- 🎯 RFM customer segmentation 
- 🛒 Market basket analysis
- 📈 Business performance metrics
- 🔍 Vietnamese language support

#### 2. Upcoming Notebooks:

**`ml_models_training.ipynb`** (Coming soon)
- Machine Learning model development
- ALS Collaborative Filtering
- Content-based recommendations
- Model evaluation và tuning

**`real_time_analytics.ipynb`** (Coming soon)  
- Kafka streaming data processing
- Real-time recommendation serving
- Performance monitoring

**`advanced_analytics.ipynb`** (Coming soon)
- Advanced statistical analysis
- Cohort analysis
- Customer lifetime value prediction
- Churn prediction models

### 📁 Project Structure for Notebooks

```
notebooks/
├── ecommerce_analysis.ipynb     # ✅ Available - Main analysis
├── ml_models_training.ipynb     # 🔄 Coming soon
├── real_time_analytics.ipynb   # 🔄 Coming soon  
├── advanced_analytics.ipynb    # 🔄 Coming soon
├── data/                        # Notebook-specific data
│   ├── processed/              # Processed datasets for analysis
│   └── results/               # Analysis results and exports
└── outputs/                    # Generated reports and charts
    ├── charts/                # Exported visualizations
    └── reports/              # Generated PDF/HTML reports
```

### 🎯 Learning Outcomes

Sau khi hoàn thành notebook analysis, bạn sẽ nắm vững:

#### Big Data Concepts:
- Data processing với pandas và Spark
- Handling large datasets efficiently
- Data visualization best practices
- Statistical analysis techniques

#### Business Analytics:
- Customer segmentation strategies
- Product performance analysis
- Market basket analysis applications
- KPI measurement và interpretation

#### Machine Learning:
- Recommendation system foundations
- Collaborative filtering concepts
- Content-based filtering approaches
- Evaluation metrics for recommendations

#### Vietnamese E-commerce Insights:
- Local market characteristics
- Vietnamese customer behavior patterns
- Cultural factors in product recommendations
- Localization strategies

### 🔧 Technical Requirements

**Software:**
- Python 3.8+
- Jupyter Notebook hoặc VS Code với Python extension
- Web browser (Chrome, Firefox, Safari)

**Python Libraries:**
- pandas, numpy - Data manipulation
- matplotlib, seaborn, plotly - Visualization  
- scikit-learn - Machine learning
- underthesea, pyvi - Vietnamese text processing

**Hardware Recommendations:**
- RAM: 8GB+ (16GB recommended for large datasets)
- CPU: Multi-core processor
- Storage: 5GB+ free space

### 📚 Additional Resources

**Documentation:**
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

**Vietnamese NLP:**
- [Underthesea Documentation](https://underthesea.readthedocs.io/)
- [PyVi GitHub Repository](https://github.com/trungtv/pyvi)

**Big Data Learning:**
- [Coursera Big Data Specialization](https://www.coursera.org/specializations/big-data)
- [edX Data Science Courses](https://www.edx.org/course/subject/data-science)

### 🆘 Troubleshooting

**Common Issues:**

1. **Memory Error:**
   ```python
   # Reduce dataset size for analysis
   sample_transactions = transactions_df.sample(n=10000)
   ```

2. **Visualization Issues:**
   ```python
   # Use static plots instead of interactive ones
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   # Your plotting code
   ```

3. **Vietnamese Font Issues:**
   ```python
   # Install Vietnamese fonts or use English labels
   plt.rcParams['font.family'] = ['Arial', 'sans-serif']
   ```

4. **Package Installation:**
   ```bash
   # Use pip with --user flag if permission issues
   pip install --user package_name
   ```

### 🎓 Assignment Guidelines

**For HUIT Students:**

1. **Required Deliverables:**
   - Completed ecommerce_analysis.ipynb với all cells executed
   - Custom analysis section với your own insights
   - Vietnamese documentation for all findings
   - Performance comparison với different algorithms

2. **Evaluation Criteria:**
   - Code quality và documentation (25%)
   - Data analysis depth và accuracy (35%)
   - Business insights và recommendations (25%)
   - Presentation và visualization (15%)

3. **Bonus Points:**
   - Additional visualizations
   - Creative analysis approaches
   - Integration with web demo
   - Performance optimization

### 📞 Support

**Getting Help:**
- Check project README.md for general setup issues
- Review notebook comments và markdown cells
- Search Stack Overflow for technical problems
- Contact course instructor for academic questions

---

**Happy Analyzing! 🚀📊**

*Chúc các bạn học tốt môn Big Data Analytics tại HUIT!*