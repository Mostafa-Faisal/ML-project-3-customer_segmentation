# ML-project-3-customer_segmentation# 🎯 Customer Segmentation ML Project

A comprehensive machine learning project that implements K-Means clustering to segment customers based on their purchasing behavior and demographics. This project includes data analysis, model development, and interactive web applications for real-time customer segmentation predictions.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Business Applications](#business-applications)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Project Overview

This project implements an end-to-end customer segmentation solution using unsupervised machine learning techniques. The system analyzes customer data to identify distinct behavioral patterns and groups customers into meaningful segments for targeted marketing and business strategy.

### Key Objectives:
- Segment customers based on purchasing behavior, demographics, and engagement patterns
- Provide actionable business insights for marketing teams
- Create an interactive web application for real-time predictions
- Enable batch processing for large customer datasets

## ✨ Features

### 🔍 Data Analysis & Processing
- **RFM Analysis**: Recency, Frequency, and Monetary value analysis
- **Feature Engineering**: Creation of meaningful customer attributes
- **Data Preprocessing**: Scaling, cleaning, and transformation
- **Exploratory Data Analysis**: Comprehensive data visualization

### 🤖 Machine Learning
- **K-Means Clustering**: Optimal cluster identification using elbow method
- **PCA Visualization**: Principal component analysis for data visualization
- **Model Persistence**: Trained models saved for production use
- **Performance Metrics**: Silhouette score and inertia evaluation

### 🖥️ Interactive Applications
- **Streamlit Web App**: Real-time customer segmentation predictions
- **Batch Processing**: Upload CSV files for bulk predictions
- **Visualization Dashboard**: Interactive charts and cluster analysis
- **Customer Profiling**: Detailed segment interpretations

## 📊 Dataset

The project works with retail customer data containing the following features:

### Input Features:
- **Age**: Customer age (18-80 years)
- **Annual Income**: Yearly income in USD
- **Spending Score**: Customer spending behavior score (1-100)
- **Purchase Frequency**: Number of purchases per period
- **Average Order Value**: Mean transaction amount
- **Years Customer**: Customer tenure in years

### Derived Features:
- **RFM Metrics**: Recency, Frequency, Monetary analysis
- **Customer Lifetime Value**: Calculated based on purchase history
- **Purchase Rate**: Frequency per day ratio

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/Mostafa-Faisal/ML-project-3-customer_segmentation.git
cd ML-project-3-customer_segmentation
```

2. **Install required dependencies:**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib
```

3. **Verify model files exist:**
```bash
ls models/
# Should show: kmeans_customer_segmentation.pkl, pca_model.pkl, scaler.pkl
```

## 💻 Usage

### 🖥️ Web Application
Launch the interactive Streamlit application:
```bash
streamlit run customer_segmentation_app.py
```

Features:
- **Single Customer Prediction**: Input customer details via sidebar
- **Batch Processing**: Upload CSV files for multiple predictions
- **Visualization**: Interactive charts showing cluster distributions
- **Customer Profiling**: Detailed segment interpretations

### 📓 Jupyter Notebooks
Explore the analysis and model development:

1. **Main Analysis**: `customer_segmentation.ipynb`
2. **Advanced Implementation**: `task2.ipynb`
3. **Basic Tutorial**: `Customer_Segmentation 1.ipynb`

### 🐍 Python API
Use the trained models programmatically:

```python
import joblib
import pandas as pd

# Load models
kmeans_model = joblib.load('models/kmeans_customer_segmentation.pkl')
scaler = joblib.load('models/scaler.pkl')

# Predict customer segment
customer_data = {
    'Age': 35,
    'Annual_Income': 60000,
    'Spending_Score': 75,
    'Purchase_Frequency': 20,
    'Average_Order_Value': 150,
    'Years_Customer': 3
}

# Make prediction
df = pd.DataFrame([customer_data])
X_scaled = scaler.transform(df)
cluster = kmeans_model.predict(X_scaled)[0]
print(f"Customer belongs to cluster: {cluster}")
```

## 📈 Model Performance

### Clustering Metrics:
- **Optimal Clusters**: Determined using elbow method and silhouette analysis
- **Silhouette Score**: >0.5 (indicating good cluster separation)
- **Feature Importance**: 6 key features for segmentation
- **Scalability**: Handles datasets with 1000+ customers efficiently

### Validation:
- **Cross-validation**: Consistent cluster assignments
- **Stability Testing**: Robust performance across different data samples
- **Business Validation**: Segments align with marketing expectations

## 📁 Project Structure

```
ML-project-3-customer_segmentation/
├── README.md                          # Project documentation
├── customer_segmentation_app.py       # Streamlit web application
├── customer_segmentation.ipynb        # Main analysis notebook
├── Customer_Segmentation 1.ipynb      # Basic tutorial notebook
├── task2.ipynb                        # Advanced implementation
├── Online Retail.xlsx                 # Sample dataset
├── segmented_customers.csv            # Output with cluster assignments
├── models/                            # Trained models directory
│   ├── kmeans_customer_segmentation.pkl
│   ├── pca_model.pkl
│   └── scaler.pkl
├── submit/                            # Project deliverables
│   ├── Screenshot_1.png
│   └── Screenshot_2.png
└── plagarizom/                        # Documentation
    └── paper.md
```

## 🛠️ Technologies Used

### Machine Learning:
- **scikit-learn**: K-Means clustering, PCA, preprocessing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Visualization:
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts

### Web Application:
- **Streamlit**: Interactive web interface
- **joblib**: Model serialization

### Development:
- **Jupyter**: Notebook development environment
- **Python 3.8+**: Core programming language

## 📊 Results

### Customer Segments Identified:

1. **💰 High-Value Customers** (Cluster 1)
   - High income and spending patterns
   - Premium product preferences
   - Target for luxury offerings

2. **🔄 Frequent Buyers** (Cluster 2)
   - Regular purchase behavior
   - Consistent engagement
   - Ideal for loyalty programs

3. **🎯 Budget-Conscious** (Cluster 0)
   - Price-sensitive customers
   - Moderate spending patterns
   - Target for discounts and promotions

4. **⭐ Premium Customers** (Cluster 3)
   - High order values
   - Quality-focused purchasing
   - VIP treatment candidates

5. **🌱 Growth Potential** (Cluster 4)
   - New or occasional customers
   - Opportunity for engagement
   - Focus on retention strategies

## 💼 Business Applications

### Marketing Strategy:
- **Targeted Campaigns**: Customize messaging per segment
- **Product Recommendations**: Personalized offerings
- **Pricing Optimization**: Segment-specific pricing strategies

### Customer Retention:
- **Loyalty Programs**: Reward frequent buyers
- **Risk Assessment**: Identify churning customers
- **Engagement Strategies**: Personalized communication

### Business Intelligence:
- **Market Analysis**: Understanding customer base composition
- **Revenue Optimization**: Focus on high-value segments
- **Resource Allocation**: Efficient marketing spend distribution

### Operational Insights:
- **Inventory Management**: Stock products based on segment preferences
- **Customer Service**: Tailored support approaches
- **Growth Planning**: Identify expansion opportunities

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Implement improvements or fixes
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**: Describe your changes

### Areas for Contribution:
- Additional clustering algorithms (DBSCAN, Hierarchical)
- Enhanced visualization features
- API development for model serving
- Mobile application development
- Performance optimizations

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Retail customer data for segmentation analysis
- Libraries: scikit-learn, pandas, streamlit communities
- Inspiration: Real-world customer analytics use cases

## 📞 Contact

**Author**: Mostafa Faisal  
**Repository**: [ML-project-3-customer_segmentation](https://github.com/Mostafa-Faisal/ML-project-3-customer_segmentation)

For questions or suggestions, please open an issue or contact through GitHub.

---

⭐ **If you found this project helpful, please give it a star!** ⭐
