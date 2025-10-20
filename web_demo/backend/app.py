"""
Flask API Backend for E-commerce Recommendation System
Author: HUIT Big Data Project
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pymongo
import redis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from pyspark.sql import SparkSession
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/ecommerce')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    SPARK_MASTER = os.getenv('SPARK_MASTER', 'local[*]')
    MODEL_PATH = os.getenv('MODEL_PATH', '/data/models')

# Initialize connections
def init_connections():
    """Initialize database connections"""
    try:
        # MongoDB connection
        mongo_client = pymongo.MongoClient(Config.MONGODB_URI)
        mongo_db = mongo_client.ecommerce
        
        # Redis connection
        redis_client = redis.from_url(Config.REDIS_URL)
        
        # Spark session for ML models
        spark = SparkSession.builder \
            .appName("RecommendationAPI") \
            .master(Config.SPARK_MASTER) \
            .getOrCreate()
        
        logger.info("Database connections initialized successfully")
        return mongo_db, redis_client, spark
        
    except Exception as e:
        logger.error(f"Error initializing connections: {str(e)}")
        raise

# Global variables
mongo_db, redis_client, spark = init_connections()

class RecommendationService:
    """Service class for handling recommendations"""
    
    def __init__(self):
        self.mongo_db = mongo_db
        self.redis_client = redis_client
        self.spark = spark
        
    def get_customer_profile(self, customer_id):
        """Get customer profile from MongoDB"""
        try:
            customer = self.mongo_db.customers.find_one({"customer_id": customer_id})
            if customer:
                # Remove MongoDB ObjectId for JSON serialization
                customer.pop('_id', None)
                return customer
            return None
        except Exception as e:
            logger.error(f"Error getting customer profile: {str(e)}")
            return None
    
    def get_customer_history(self, customer_id, limit=50):
        """Get customer purchase history"""
        try:
            history = list(self.mongo_db.transactions.find(
                {"customer_id": customer_id}
            ).sort("timestamp", -1).limit(limit))
            
            # Remove MongoDB ObjectIds
            for item in history:
                item.pop('_id', None)
            
            return history
        except Exception as e:
            logger.error(f"Error getting customer history: {str(e)}")
            return []
    
    def get_product_info(self, product_id):
        """Get product information"""
        try:
            product = self.mongo_db.products.find_one({"product_id": product_id})
            if product:
                product.pop('_id', None)
                return product
            return None
        except Exception as e:
            logger.error(f"Error getting product info: {str(e)}")
            return None
    
    def get_trending_products(self, limit=20, days=7):
        """Get trending products based on recent sales"""
        try:
            # Get products sold in last N days
            start_date = datetime.now() - timedelta(days=days)
            
            pipeline = [
                {"$match": {"timestamp": {"$gte": start_date}}},
                {"$group": {
                    "_id": "$product_id",
                    "total_sales": {"$sum": "$quantity"},
                    "total_revenue": {"$sum": "$total_amount"},
                    "unique_customers": {"$addToSet": "$customer_id"}
                }},
                {"$addFields": {
                    "customer_count": {"$size": "$unique_customers"}
                }},
                {"$sort": {"total_sales": -1}},
                {"$limit": limit}
            ]
            
            trending = list(self.mongo_db.transactions.aggregate(pipeline))
            
            # Enrich with product details
            for item in trending:
                product_info = self.get_product_info(item["_id"])
                if product_info:
                    item.update(product_info)
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending products: {str(e)}")
            return []
    
    def get_collaborative_recommendations(self, customer_id, limit=10):
        """Get collaborative filtering recommendations"""
        try:
            # Check Redis cache first
            cache_key = f"cf_recommendations:{customer_id}"
            cached = self.redis_client.get(cache_key)
            
            if cached:
                return json.loads(cached)
            
            # Generate recommendations (simplified version)
            # In production, this would use the trained ML model
            customer_history = self.get_customer_history(customer_id)
            if not customer_history:
                return []
            
            # Get products from same categories as customer's history
            purchased_categories = list(set([item.get('category', '') for item in customer_history]))
            
            pipeline = [
                {"$match": {"category": {"$in": purchased_categories}}},
                {"$sample": {"size": limit * 2}}
            ]
            
            candidates = list(self.mongo_db.products.aggregate(pipeline))
            
            # Filter out already purchased products
            purchased_products = [item['product_id'] for item in customer_history]
            recommendations = [
                product for product in candidates 
                if product['product_id'] not in purchased_products
            ][:limit]
            
            # Remove MongoDB ObjectIds
            for rec in recommendations:
                rec.pop('_id', None)
            
            # Cache for 1 hour
            self.redis_client.setex(
                cache_key, 
                3600, 
                json.dumps(recommendations, default=str)
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting collaborative recommendations: {str(e)}")
            return []
    
    def get_content_recommendations(self, product_id, limit=10):
        """Get content-based recommendations for a product"""
        try:
            cache_key = f"content_recommendations:{product_id}"
            cached = self.redis_client.get(cache_key)
            
            if cached:
                return json.loads(cached)
            
            # Get product info
            target_product = self.get_product_info(product_id)
            if not target_product:
                return []
            
            # Find similar products in same category and price range
            price = target_product.get('price', 0)
            category = target_product.get('category', '')
            
            # Define price range (±20%)
            price_min = price * 0.8
            price_max = price * 1.2
            
            similar_products = list(self.mongo_db.products.find({
                "category": category,
                "price": {"$gte": price_min, "$lte": price_max},
                "product_id": {"$ne": product_id}
            }).limit(limit))
            
            # Remove MongoDB ObjectIds
            for product in similar_products:
                product.pop('_id', None)
            
            # Cache for 2 hours
            self.redis_client.setex(
                cache_key, 
                7200, 
                json.dumps(similar_products, default=str)
            )
            
            return similar_products
            
        except Exception as e:
            logger.error(f"Error getting content recommendations: {str(e)}")
            return []
    
    def search_products(self, query, category=None, price_min=None, price_max=None, limit=20):
        """Search products with filters"""
        try:
            # Build search criteria
            search_criteria = {}
            
            # Text search
            if query:
                search_criteria["$or"] = [
                    {"product_name": {"$regex": query, "$options": "i"}},
                    {"description": {"$regex": query, "$options": "i"}},
                    {"brand": {"$regex": query, "$options": "i"}}
                ]
            
            # Category filter
            if category:
                search_criteria["category"] = category
            
            # Price range filter
            price_filter = {}
            if price_min is not None:
                price_filter["$gte"] = float(price_min)
            if price_max is not None:
                price_filter["$lte"] = float(price_max)
            if price_filter:
                search_criteria["price"] = price_filter
            
            # Execute search
            products = list(self.mongo_db.products.find(search_criteria).limit(limit))
            
            # Remove MongoDB ObjectIds
            for product in products:
                product.pop('_id', None)
            
            return products
            
        except Exception as e:
            logger.error(f"Error searching products: {str(e)}")
            return []

# Initialize service
recommendation_service = RecommendationService()

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/customer/<customer_id>/profile', methods=['GET'])
def get_customer_profile(customer_id):
    """Get customer profile"""
    profile = recommendation_service.get_customer_profile(customer_id)
    if profile:
        return jsonify({"success": True, "data": profile})
    return jsonify({"success": False, "message": "Customer not found"}), 404

@app.route('/api/customer/<customer_id>/history', methods=['GET'])
def get_customer_history(customer_id):
    """Get customer purchase history"""
    limit = request.args.get('limit', 50, type=int)
    history = recommendation_service.get_customer_history(customer_id, limit)
    return jsonify({"success": True, "data": history})

@app.route('/api/customer/<customer_id>/recommendations', methods=['GET'])
def get_recommendations(customer_id):
    """Get personalized recommendations for customer"""
    rec_type = request.args.get('type', 'collaborative')
    limit = request.args.get('limit', 10, type=int)
    
    if rec_type == 'collaborative':
        recommendations = recommendation_service.get_collaborative_recommendations(customer_id, limit)
    else:
        recommendations = []
    
    return jsonify({"success": True, "data": recommendations})

@app.route('/api/product/<product_id>/similar', methods=['GET'])
def get_similar_products(product_id):
    """Get similar products (content-based)"""
    limit = request.args.get('limit', 10, type=int)
    similar = recommendation_service.get_content_recommendations(product_id, limit)
    return jsonify({"success": True, "data": similar})

@app.route('/api/products/trending', methods=['GET'])
def get_trending():
    """Get trending products"""
    limit = request.args.get('limit', 20, type=int)
    days = request.args.get('days', 7, type=int)
    trending = recommendation_service.get_trending_products(limit, days)
    return jsonify({"success": True, "data": trending})

@app.route('/api/products/search', methods=['GET'])
def search_products():
    """Search products"""
    query = request.args.get('q', '')
    category = request.args.get('category')
    price_min = request.args.get('price_min')
    price_max = request.args.get('price_max')
    limit = request.args.get('limit', 20, type=int)
    
    results = recommendation_service.search_products(
        query, category, price_min, price_max, limit
    )
    return jsonify({"success": True, "data": results})

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all product categories"""
    try:
        categories = mongo_db.products.distinct("category")
        return jsonify({"success": True, "data": categories})
    except Exception as e:
        logger.error(f"Error getting categories: {str(e)}")
        return jsonify({"success": False, "message": "Error getting categories"}), 500

@app.route('/api/analytics/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get dashboard analytics data"""
    try:
        # Total customers
        total_customers = mongo_db.customers.count_documents({})
        
        # Total products
        total_products = mongo_db.products.count_documents({})
        
        # Total transactions (last 30 days)
        start_date = datetime.now() - timedelta(days=30)
        total_transactions = mongo_db.transactions.count_documents({
            "timestamp": {"$gte": start_date}
        })
        
        # Total revenue (last 30 days)
        revenue_pipeline = [
            {"$match": {"timestamp": {"$gte": start_date}}},
            {"$group": {"_id": None, "total": {"$sum": "$total_amount"}}}
        ]
        revenue_result = list(mongo_db.transactions.aggregate(revenue_pipeline))
        total_revenue = revenue_result[0]["total"] if revenue_result else 0
        
        # Top categories
        category_pipeline = [
            {"$match": {"timestamp": {"$gte": start_date}}},
            {"$group": {
                "_id": "$category",
                "sales": {"$sum": "$quantity"},
                "revenue": {"$sum": "$total_amount"}
            }},
            {"$sort": {"sales": -1}},
            {"$limit": 5}
        ]
        top_categories = list(mongo_db.transactions.aggregate(category_pipeline))
        
        dashboard_data = {
            "totals": {
                "customers": total_customers,
                "products": total_products,
                "transactions": total_transactions,
                "revenue": total_revenue
            },
            "top_categories": top_categories
        }
        
        return jsonify({"success": True, "data": dashboard_data})
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({"success": False, "message": "Error getting dashboard data"}), 500

# Demo web interface routes
@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard page"""
    return render_template('dashboard.html')

@app.route('/customer/<customer_id>')
def customer_page(customer_id):
    """Customer-specific recommendation page"""
    return render_template('customer.html', customer_id=customer_id)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "message": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)