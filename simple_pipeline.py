#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Pipeline Runner for HUIT Big Data Project
Chạy pipeline cơ bản không cần Spark và các thư viện phức tạp
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def setup_directories():
    """Tạo các thư mục cần thiết"""
    dirs = [
        'data/sample',
        'data/processed', 
        'logs',
        'web_demo/static/css',
        'web_demo/static/js',
        'web_demo/templates'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(project_root, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"✅ Created directory: {full_path}")

def generate_sample_data():
    """Tạo dữ liệu mẫu đơn giản"""
    
    # Kiểm tra xem data đã tồn tại chưa
    data_dir = os.path.join(project_root, 'data', 'sample')
    customers_file = os.path.join(data_dir, 'customers.csv')
    products_file = os.path.join(data_dir, 'products.csv')
    transactions_file = os.path.join(data_dir, 'transactions.csv')
    
    if all(os.path.exists(f) for f in [customers_file, products_file, transactions_file]):
        print("📋 Sample data already exists, loading existing data...")
        customers_df = pd.read_csv(customers_file)
        products_df = pd.read_csv(products_file)
        transactions_df = pd.read_csv(transactions_file)
        
        print(f"✅ Loaded {len(customers_df)} customers")
        print(f"✅ Loaded {len(products_df)} products")
        print(f"✅ Loaded {len(transactions_df)} transactions")
        
        return customers_df, products_df, transactions_df
    
    print("🔄 Generating sample data...")
    
    # Tạo customers
    customers = []
    cities = ['Hồ Chí Minh', 'Hà Nội', 'Đà Nẵng', 'Cần Thơ', 'Hải Phòng', 'Biên Hòa', 'Nha Trang', 'Huế']
    
    for i in range(1, 1001):
        customer = {
            'customer_id': f'CUST_{i:06d}',
            'name': f'Khách hàng {i}',
            'email': f'customer{i}@email.com',
            'age': random.randint(18, 65),
            'gender': random.choice(['Nam', 'Nữ']),
            'city': random.choice(cities),
            'registration_date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
        }
        customers.append(customer)
    
    customers_df = pd.DataFrame(customers)
    
    # Tạo products
    products = []
    categories = ['Điện thoại', 'Laptop', 'Tivi', 'Quần áo', 'Giày dép', 'Túi xách', 'Đồng hồ', 'Phụ kiện']
    
    for i in range(1, 501):
        product = {
            'product_id': f'PROD_{i:06d}',
            'product_name': f'Sản phẩm {i}',
            'category': random.choice(categories),
            'price': random.randint(100000, 50000000),
            'brand': f'Brand {random.randint(1, 20)}',
            'rating': round(random.uniform(3.0, 5.0), 1)
        }
        products.append(product)
    
    products_df = pd.DataFrame(products)
    
    # Tạo transactions
    transactions = []
    for i in range(1, 10001):
        customer_id = random.choice(customers_df['customer_id'])
        product = products_df.iloc[random.randint(0, len(products_df)-1)]
        quantity = random.randint(1, 5)
        
        transaction = {
            'transaction_id': f'TXN_{i:08d}',
            'customer_id': customer_id,
            'product_id': product['product_id'],
            'category': product['category'],
            'quantity': quantity,
            'unit_price': product['price'],
            'total_amount': quantity * product['price'],
            'timestamp': (datetime.now() - timedelta(days=random.randint(1, 30), 
                                                   hours=random.randint(0, 23), 
                                                   minutes=random.randint(0, 59))).strftime('%Y-%m-%d %H:%M:%S')
        }
        transactions.append(transaction)
    
    transactions_df = pd.DataFrame(transactions)
    
    # Save data
    data_dir = os.path.join(project_root, 'data', 'sample')
    customers_df.to_csv(os.path.join(data_dir, 'customers.csv'), index=False, encoding='utf-8')
    products_df.to_csv(os.path.join(data_dir, 'products.csv'), index=False, encoding='utf-8')
    transactions_df.to_csv(os.path.join(data_dir, 'transactions.csv'), index=False, encoding='utf-8')
    
    print(f"✅ Generated {len(customers_df)} customers")
    print(f"✅ Generated {len(products_df)} products") 
    print(f"✅ Generated {len(transactions_df)} transactions")
    
    return customers_df, products_df, transactions_df

def basic_analytics(customers_df, products_df, transactions_df):
    """Thực hiện phân tích cơ bản"""
    print("📊 Running basic analytics...")
    
    # Basic statistics
    total_revenue = transactions_df['total_amount'].sum()
    avg_order_value = transactions_df['total_amount'].mean()
    total_customers = len(customers_df)
    active_customers = transactions_df['customer_id'].nunique()
    
    # Top categories
    category_sales = transactions_df.groupby('category')['total_amount'].sum().sort_values(ascending=False)
    
    # Customer analysis
    customer_stats = transactions_df.groupby('customer_id').agg({
        'total_amount': 'sum',
        'transaction_id': 'count'
    }).sort_values('total_amount', ascending=False)
    
    # Analytics results
    analytics = {
        'total_revenue': int(total_revenue),
        'avg_order_value': int(avg_order_value),
        'total_customers': total_customers,
        'active_customers': active_customers,
        'top_categories': category_sales.head(5).to_dict(),
        'top_customers': customer_stats.head(10).to_dict('index'),
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save analytics
    analytics_file = os.path.join(project_root, 'data', 'processed', 'analytics_results.json')
    with open(analytics_file, 'w', encoding='utf-8') as f:
        json.dump(analytics, f, ensure_ascii=False, indent=2)
    
    print(f"💰 Total Revenue: {total_revenue:,.0f} VND")
    print(f"📈 Average Order Value: {avg_order_value:,.0f} VND")
    print(f"👥 Active Customers: {active_customers}/{total_customers}")
    print(f"🏆 Top Category: {category_sales.index[0]} ({category_sales.iloc[0]:,.0f} VND)")
    
    return analytics

def create_simple_web_demo():
    """Tạo web demo đơn giản"""
    print("🌐 Creating simple web demo...")
    
    # Simple Flask app
    app_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Web Demo for HUIT Big Data Project
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, jsonify, request
from datetime import datetime

app = Flask(__name__)

# Load data
def load_data():
    try:
        customers_df = pd.read_csv('data/sample/customers.csv')
        products_df = pd.read_csv('data/sample/products.csv') 
        transactions_df = pd.read_csv('data/sample/transactions.csv')
        
        with open('data/processed/analytics_results.json', 'r', encoding='utf-8') as f:
            analytics = json.load(f)
            
        return customers_df, products_df, transactions_df, analytics
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, {}

@app.route('/')
def dashboard():
    """Main dashboard"""
    customers_df, products_df, transactions_df, analytics = load_data()
    
    if customers_df is None:
        return "<h1>❌ Error: Data not found. Please run the pipeline first!</h1>"
    
    return render_template('dashboard.html', 
                         analytics=analytics,
                         total_customers=len(customers_df),
                         total_products=len(products_df),
                         total_transactions=len(transactions_df))

@app.route('/api/analytics')
def api_analytics():
    """API endpoint for analytics data"""
    _, _, _, analytics = load_data()
    return jsonify(analytics)

@app.route('/api/recommendations/<customer_id>')
def api_recommendations(customer_id):
    """Simple recommendation API"""
    customers_df, products_df, transactions_df, _ = load_data()
    
    if customers_df is None:
        return jsonify({'error': 'Data not available'})
    
    # Simple recommendation: top products by category
    customer_transactions = transactions_df[transactions_df['customer_id'] == customer_id]
    
    if customer_transactions.empty:
        # New customer: recommend popular products
        popular_products = transactions_df.groupby('product_id').size().head(5).index.tolist()
    else:
        # Existing customer: recommend from favorite category
        favorite_category = customer_transactions['category'].mode().iloc[0] if not customer_transactions['category'].mode().empty else 'Điện thoại'
        category_products = products_df[products_df['category'] == favorite_category]
        popular_products = category_products.head(5)['product_id'].tolist()
    
    recommendations = []
    for product_id in popular_products:
        product = products_df[products_df['product_id'] == product_id].iloc[0] if not products_df[products_df['product_id'] == product_id].empty else None
        if product is not None:
            recommendations.append({
                'product_id': product['product_id'],
                'product_name': product['product_name'],
                'price': int(product['price']),
                'category': product['category']
            })
    
    return jsonify({'customer_id': customer_id, 'recommendations': recommendations})

if __name__ == '__main__':
    print("🚀 Starting HUIT Big Data Web Demo...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔗 API: http://localhost:5000/api/analytics")
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    app_file = os.path.join(project_root, 'simple_app.py')
    with open(app_file, 'w', encoding='utf-8') as f:
        f.write(app_content)
    
    # Simple HTML template
    template_content = '''<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HUIT Big Data Project - E-commerce Analytics</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <div class="container-fluid">
        <header class="bg-primary text-white p-3 mb-4">
            <h1><i class="fas fa-chart-line"></i> HUIT Big Data Project</h1>
            <p>Phân tích Dữ liệu Mua sắm Trực tuyến & Hệ thống Đề xuất Sản phẩm</p>
        </header>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5><i class="fas fa-users text-primary"></i></h5>
                        <h3>{{ total_customers:,d }}</h3>
                        <p>Tổng Khách hàng</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5><i class="fas fa-box text-success"></i></h5>
                        <h3>{{ total_products:,d }}</h3>
                        <p>Tổng Sản phẩm</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5><i class="fas fa-shopping-cart text-warning"></i></h5>
                        <h3>{{ total_transactions:,d }}</h3>
                        <p>Tổng Giao dịch</p>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5><i class="fas fa-money-bill-wave text-info"></i></h5>
                        <h3>{{ "{:,}".format(analytics.total_revenue) if analytics.total_revenue else "0" }}</h3>
                        <p>Doanh thu (VND)</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-chart-pie"></i> Top Categories</h5>
                    </div>
                    <div class="card-body">
                        {% if analytics.top_categories %}
                            {% for category, revenue in analytics.top_categories.items() %}
                            <div class="d-flex justify-content-between">
                                <span>{{ category }}</span>
                                <strong>{{ "{:,}".format(revenue) }} VND</strong>
                            </div>
                            <hr>
                            {% endfor %}
                        {% endif %}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-star"></i> Analytics Summary</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Average Order Value:</strong> {{ "{:,}".format(analytics.avg_order_value) if analytics.avg_order_value else "0" }} VND</p>
                        <p><strong>Active Customers:</strong> {{ analytics.active_customers if analytics.active_customers else 0 }}</p>
                        <p><strong>Generated:</strong> {{ analytics.generated_at if analytics.generated_at else "Unknown" }}</p>
                        <div class="mt-3">
                            <a href="/api/analytics" class="btn btn-primary">
                                <i class="fas fa-download"></i> Download Analytics JSON
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5><i class="fas fa-robot"></i> Recommendation System Demo</h5>
                    </div>
                    <div class="card-body">
                        <div class="input-group mb-3">
                            <input type="text" class="form-control" id="customerInput" placeholder="Enter Customer ID (e.g., CUST_000001)">
                            <button class="btn btn-success" onclick="getRecommendations()">
                                <i class="fas fa-search"></i> Get Recommendations
                            </button>
                        </div>
                        <div id="recommendations"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    function getRecommendations() {
        const customerId = document.getElementById('customerInput').value;
        if (!customerId) {
            alert('Please enter a Customer ID');
            return;
        }
        
        fetch(`/api/recommendations/${customerId}`)
            .then(response => response.json())
            .then(data => {
                let html = '<h6>Recommendations for ' + data.customer_id + ':</h6>';
                if (data.recommendations && data.recommendations.length > 0) {
                    html += '<div class="row">';
                    data.recommendations.forEach(product => {
                        html += `
                            <div class="col-md-4 mb-2">
                                <div class="card">
                                    <div class="card-body">
                                        <h6>${product.product_name}</h6>
                                        <p>Category: ${product.category}</p>
                                        <strong>${product.price.toLocaleString()} VND</strong>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    html += '</div>';
                } else {
                    html += '<p>No recommendations available.</p>';
                }
                document.getElementById('recommendations').innerHTML = html;
            })
            .catch(error => {
                document.getElementById('recommendations').innerHTML = '<p class="text-danger">Error loading recommendations</p>';
            });
    }
    </script>
</body>
</html>'''
    
    template_dir = os.path.join(project_root, 'web_demo', 'templates')
    template_file = os.path.join(template_dir, 'dashboard.html')
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print("✅ Created simple web application")
    
def main():
    """Main pipeline function"""
    print("🚀 HUIT Big Data Project - Simple Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Setup directories
        setup_directories()
        
        # Step 2: Generate data
        customers_df, products_df, transactions_df = generate_sample_data()
        
        # Step 3: Run analytics
        analytics = basic_analytics(customers_df, products_df, transactions_df)
        
        # Step 4: Create web demo
        create_simple_web_demo()
        
        print("\n" + "=" * 50)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print("\n🎯 NEXT STEPS:")
        print("1. Run web demo: python simple_app.py")
        print("2. Open browser: http://localhost:5000")
        print("3. Try Jupyter notebook: jupyter notebook notebooks/ecommerce_analysis.ipynb")
        print("4. Check data: data/sample/*.csv")
        
        print("\n📊 QUICK STATS:")
        print(f"• Total Revenue: {analytics['total_revenue']:,} VND")
        print(f"• Active Customers: {analytics['active_customers']}")
        print(f"• Top Category: {list(analytics['top_categories'].keys())[0] if analytics['top_categories'] else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()