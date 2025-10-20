#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Web Demo for HUIT Big Data Project
"""

import os
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from datetime import datetime
import io

app = Flask(__name__, template_folder='web_demo/templates')
app.secret_key = 'huit_bigdata_2025'
app.config['UPLOAD_FOLDER'] = 'data/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'txt', 'xlsx', 'json'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_uploaded_file(filepath, filename):
    """Analyze uploaded CSV/TXT file - Optimized for Kaggle datasets"""
    try:
        # Determine file type and read accordingly
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'csv':
            # Try different encodings and separators for CSV (Kaggle compatibility)
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
            separators = [',', ';', '\t', '|']
            df = None
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        # Try to read with different parameters for Kaggle datasets
                        df = pd.read_csv(filepath, encoding=encoding, sep=sep, 
                                       low_memory=False, na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NaN'])
                        
                        # Validate that we have reasonable data
                        if df.shape[1] > 1 and df.shape[0] > 0:
                            break
                    except:
                        continue
                if df is not None and df.shape[1] > 1:
                    break
            
            if df is None or df.shape[1] <= 1:
                return {'error': 'Cannot read CSV file with supported encodings and separators'}
                
        elif file_ext == 'txt':
            # Read as text and try to convert to DataFrame
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to detect delimiter
            lines = content.strip().split('\n')
            if len(lines) > 1:
                # Check for common delimiters
                delimiters = [',', ';', '\t', '|']
                for delimiter in delimiters:
                    if delimiter in lines[0]:
                        try:
                            df = pd.read_csv(io.StringIO(content), sep=delimiter)
                            break
                        except:
                            continue
                else:
                    # If no delimiter found, treat as single column
                    df = pd.DataFrame({'text': lines})
            else:
                df = pd.DataFrame({'text': [content]})
                
        elif file_ext == 'xlsx':
            df = pd.read_excel(filepath)
            
        elif file_ext == 'json':
            df = pd.read_json(filepath)
            
        # Clean and prepare data for analysis
        # Handle common issues in Kaggle datasets
        df = df.replace([np.inf, -np.inf], np.nan)  # Replace infinity values
        
        # Convert numeric columns properly
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
        
        # Perform basic analysis with safe conversions
        summary_data = {}
        try:
            summary_raw = df.describe(include='all').fillna('N/A')
            for col in summary_raw.columns:
                summary_data[str(col)] = {}
                for stat in summary_raw.index:
                    value = summary_raw.loc[stat, col]
                    if pd.isna(value) or value == 'N/A':
                        summary_data[str(col)][str(stat)] = 'N/A'
                    elif isinstance(value, (int, float)):
                        summary_data[str(col)][str(stat)] = float(value) if not pd.isna(value) else 'N/A'
                    else:
                        summary_data[str(col)][str(stat)] = str(value)
        except Exception as e:
            summary_data = {}
        
        # Safe head data conversion
        head_data = []
        try:
            for _, row in df.head(10).iterrows():
                row_dict = {}
                for col in df.columns:
                    val = row[col]
                    if pd.isna(val):
                        row_dict[str(col)] = 'N/A'
                    elif isinstance(val, (int, float)):
                        row_dict[str(col)] = val if not pd.isna(val) else 'N/A'
                    else:
                        row_dict[str(col)] = str(val)
                head_data.append(row_dict)
        except:
            head_data = []
        
        analysis = {
            'filename': filename,
            'shape': list(df.shape),
            'columns': [str(col) for col in df.columns],
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'head': head_data,
            'summary': summary_data,
            'null_counts': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Enhanced financial analysis - detect common Kaggle financial columns
        financial_keywords = ['price', 'amount', 'revenue', 'cost', 'salary', 'income', 
                             'profit', 'sales', 'value', 'total', 'sum', 'fee', 'charge']
        
        financial_cols = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in financial_keywords):
                if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
                    financial_cols.append(col)
        
        if financial_cols:
            # Financial data analysis with safe calculations
            try:
                financial_metrics = {'total': {}, 'average': {}, 'max': {}, 'min': {}}
                
                for col in financial_cols:
                    col_data = pd.to_numeric(df[col], errors='coerce').dropna()
                    if len(col_data) > 0:
                        financial_metrics['total'][str(col)] = float(col_data.sum())
                        financial_metrics['average'][str(col)] = float(col_data.mean())
                        financial_metrics['max'][str(col)] = float(col_data.max())
                        financial_metrics['min'][str(col)] = float(col_data.min())
                
                if any(financial_metrics['total'].values()):
                    analysis['financial_metrics'] = financial_metrics
            except Exception as e:
                pass  # Skip financial analysis if error
        
        # Text analysis if text columns exist
        text_cols = df.select_dtypes(include=['object']).columns.tolist()
        if text_cols:
            analysis['text_analysis'] = {}
            for col in text_cols[:3]:  # Analyze first 3 text columns
                top_values = df[col].value_counts().head(5)
                analysis['text_analysis'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in top_values.to_dict().items()},
                    'avg_length': float(df[col].astype(str).str.len().mean()) if not df[col].isna().all() else 0.0
                }
        
        return analysis
        
    except Exception as e:
        return {'error': f'Error analyzing file: {str(e)}'}

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

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and analysis"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to avoid conflicts
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
            filename = timestamp + filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                file.save(filepath)
                
                # Analyze the file
                analysis = analyze_uploaded_file(filepath, file.filename)
                
                if 'error' in analysis:
                    flash(f'Error analyzing file: {analysis["error"]}')
                    os.remove(filepath)  # Clean up failed file
                    return redirect(request.url)
                
                # Save analysis results
                analysis_file = os.path.join(app.config['UPLOAD_FOLDER'], 
                                           filename.replace('.', '_analysis.'))
                analysis_file = analysis_file.rsplit('.', 1)[0] + '.json'
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, ensure_ascii=False, indent=2, default=str)
                
                flash(f'File uploaded and analyzed successfully: {file.filename}')
                return redirect(url_for('file_analysis', filename=filename))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('Invalid file type. Allowed: CSV, TXT, XLSX, JSON')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/analysis/<filename>')
def file_analysis(filename):
    """Display analysis results"""
    try:
        # Load analysis results
        analysis_file = os.path.join(app.config['UPLOAD_FOLDER'], 
                                   filename.replace('.', '_analysis.'))
        analysis_file = analysis_file.rsplit('.', 1)[0] + '.json'
        
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        return render_template('analysis.html', analysis=analysis)
        
    except Exception as e:
        flash(f'Error loading analysis: {str(e)}')
        return redirect(url_for('upload_file'))

@app.route('/api/upload-analysis/<filename>')
def api_upload_analysis(filename):
    """API endpoint for upload analysis data"""
    try:
        analysis_file = os.path.join(app.config['UPLOAD_FOLDER'], 
                                   filename.replace('.', '_analysis.'))
        analysis_file = analysis_file.rsplit('.', 1)[0] + '.json'
        
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/uploads')
def list_uploads():
    """List all uploaded files"""
    try:
        files = []
        upload_dir = app.config['UPLOAD_FOLDER']
        
        for filename in os.listdir(upload_dir):
            if not filename.endswith('_analysis.json'):
                filepath = os.path.join(upload_dir, filename)
                stat = os.stat(filepath)
                
                files.append({
                    'filename': filename,
                    'size': stat.st_size,
                    'upload_time': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'analysis_available': os.path.exists(filepath.replace('.', '_analysis.').rsplit('.', 1)[0] + '.json')
                })
        
        files.sort(key=lambda x: x['upload_time'], reverse=True)
        return render_template('uploads.html', files=files)
        
    except Exception as e:
        flash(f'Error listing uploads: {str(e)}')
        return render_template('uploads.html', files=[])

if __name__ == '__main__':
    print("🚀 Starting HUIT Big Data Web Demo...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔗 API: http://localhost:5000/api/analytics")
    app.run(debug=True, host='0.0.0.0', port=5000)