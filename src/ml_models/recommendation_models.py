"""
Machine Learning Models for E-commerce Recommendation System
Author: HUIT Big Data Project
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

class RecommendationModels:
    def __init__(self, spark_session=None):
        """Initialize ML models for recommendation system"""
        if spark_session is None:
            self.spark = SparkSession.builder \
                .appName("EcommerceML") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
        else:
            self.spark = spark_session
            
        self.logger = logging.getLogger(__name__)
    
    def prepare_als_data(self, transactions_df):
        """
        Prepare data for Collaborative Filtering (ALS)
        
        Args:
            transactions_df: Cleaned transactions DataFrame
            
        Returns:
            DataFrame: Prepared data for ALS model
        """
        # Aggregate ratings based on purchase frequency and amount
        user_item_ratings = transactions_df.groupBy("customer_id", "product_id").agg(
            count("transaction_id").alias("frequency"),
            sum("total_amount").alias("total_spent"),
            avg("total_amount").alias("avg_amount")
        )
        
        # Create implicit ratings (0-5 scale)
        # Higher frequency and amount = higher rating
        max_freq = user_item_ratings.select(max("frequency")).collect()[0][0]
        max_amount = user_item_ratings.select(max("total_spent")).collect()[0][0]
        
        user_item_ratings = user_item_ratings.withColumn("rating",
            ((col("frequency") / max_freq) * 2.5 + 
             (col("total_spent") / max_amount) * 2.5).cast("float")
        )
        
        # Cap ratings at 5.0
        user_item_ratings = user_item_ratings.withColumn("rating",
            when(col("rating") > 5.0, 5.0).otherwise(col("rating"))
        )
        
        # Index string IDs to numeric for ALS
        customer_indexer = StringIndexer(inputCol="customer_id", outputCol="customer_index")
        product_indexer = StringIndexer(inputCol="product_id", outputCol="product_index")
        
        # Fit indexers
        customer_model = customer_indexer.fit(user_item_ratings)
        product_model = product_indexer.fit(user_item_ratings)
        
        # Transform data
        indexed_data = customer_model.transform(user_item_ratings)
        indexed_data = product_model.transform(indexed_data)
        
        return indexed_data, customer_model, product_model
    
    def train_collaborative_filtering(self, ratings_df, rank=50, max_iter=20, reg_param=0.1):
        """
        Train Collaborative Filtering model using ALS
        
        Args:
            ratings_df: Prepared ratings DataFrame
            rank: Number of latent factors
            max_iter: Maximum iterations
            reg_param: Regularization parameter
            
        Returns:
            Trained ALS model
        """
        # Split data for training and validation
        (training, validation) = ratings_df.randomSplit([0.8, 0.2], seed=42)
        
        # Build ALS model
        als = ALS(
            maxIter=max_iter,
            regParam=reg_param,
            rank=rank,
            userCol="customer_index",
            itemCol="product_index",
            ratingCol="rating",
            coldStartStrategy="drop",
            implicitPrefs=False
        )
        
        # Train model
        model = als.fit(training)
        
        # Evaluate model
        predictions = model.transform(validation)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        
        rmse = evaluator.evaluate(predictions)
        self.logger.info(f"Collaborative Filtering RMSE: {rmse}")
        
        return model, rmse
    
    def get_user_recommendations(self, cf_model, customer_index, num_recommendations=10):
        """
        Get product recommendations for a specific user
        
        Args:
            cf_model: Trained ALS model
            customer_index: Indexed customer ID
            num_recommendations: Number of recommendations
            
        Returns:
            DataFrame: Top product recommendations
        """
        # Generate recommendations for specific user
        user_df = self.spark.createDataFrame([(customer_index,)], ["customer_index"])
        recommendations = cf_model.recommendForUserSubset(user_df, num_recommendations)
        
        # Explode recommendations
        recommendations = recommendations.select(
            "customer_index",
            explode("recommendations").alias("recommendation")
        )
        
        recommendations = recommendations.select(
            "customer_index",
            col("recommendation.product_index").alias("product_index"),
            col("recommendation.rating").alias("predicted_rating")
        )
        
        return recommendations
    
    def train_customer_segmentation(self, customer_features_df, k=5):
        """
        Train customer segmentation model using K-Means
        
        Args:
            customer_features_df: Customer features DataFrame
            k: Number of clusters
            
        Returns:
            Trained K-Means model and predictions
        """
        # Select numerical features for clustering
        feature_cols = [
            "age", "total_transactions", "total_spent", 
            "avg_order_value", "recency_days", "unique_products"
        ]
        
        # Handle missing values
        for col_name in feature_cols:
            customer_features_df = customer_features_df.fillna(0, subset=[col_name])
        
        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        feature_data = assembler.transform(customer_features_df)
        
        # Train K-Means
        kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="cluster")
        model = kmeans.fit(feature_data)
        
        # Make predictions
        predictions = model.transform(feature_data)
        
        # Evaluate clustering
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        self.logger.info(f"K-Means Silhouette Score: {silhouette}")
        
        return model, predictions, silhouette
    
    def create_content_based_features(self, products_df, transactions_df):
        """
        Create content-based features for products
        
        Args:
            products_df: Products DataFrame
            transactions_df: Transactions DataFrame
            
        Returns:
            DataFrame: Products with content features
        """
        # Calculate product popularity metrics
        product_stats = transactions_df.groupBy("product_id").agg(
            count("transaction_id").alias("popularity_score"),
            avg("total_amount").alias("avg_price_sold"),
            countDistinct("customer_id").alias("customer_reach")
        )
        
        # Join with product information
        content_features = products_df.join(product_stats, "product_id", "left")
        content_features = content_features.fillna(0, ["popularity_score", "avg_price_sold", "customer_reach"])
        
        # Create category popularity
        category_stats = content_features.groupBy("category").agg(
            avg("popularity_score").alias("category_popularity"),
            count("product_id").alias("category_size")
        )
        
        content_features = content_features.join(category_stats, "category", "left")
        
        return content_features
    
    def calculate_content_similarity(self, products_df, target_product_id):
        """
        Calculate content-based similarity for a target product
        
        Args:
            products_df: Products DataFrame with features
            target_product_id: ID of target product
            
        Returns:
            List of similar products with similarity scores
        """
        # Convert to Pandas for similarity calculation
        products_pd = products_df.toPandas()
        
        # Select numerical features for similarity
        feature_columns = ["price", "popularity_score", "category_popularity"]
        
        # Fill missing values
        for col in feature_columns:
            products_pd[col] = products_pd[col].fillna(0)
        
        # Get target product features
        target_product = products_pd[products_pd["product_id"] == target_product_id]
        if target_product.empty:
            return []
        
        target_features = target_product[feature_columns].values
        
        # Calculate similarity with all products
        all_features = products_pd[feature_columns].values
        similarities = cosine_similarity(target_features, all_features)[0]
        
        # Create similarity DataFrame
        products_pd["similarity_score"] = similarities
        
        # Filter and sort by similarity
        similar_products = products_pd[products_pd["product_id"] != target_product_id]
        similar_products = similar_products.sort_values("similarity_score", ascending=False)
        
        return similar_products[["product_id", "product_name", "category", "similarity_score"]].head(10)
    
    def train_purchase_prediction(self, customer_features_df):
        """
        Train model to predict customer purchase likelihood
        
        Args:
            customer_features_df: Customer features DataFrame
            
        Returns:
            Trained classification model
        """
        # Create target variable (will purchase in next 30 days)
        purchase_data = customer_features_df.withColumn("will_purchase",
            when(col("recency_days") <= 30, 1.0).otherwise(0.0)
        )
        
        # Select features for prediction
        feature_cols = [
            "age", "total_transactions", "avg_order_value",
            "recency_days", "unique_products", "customer_lifetime_days"
        ]
        
        # Handle missing values
        for col_name in feature_cols:
            purchase_data = purchase_data.fillna(0, subset=[col_name])
        
        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        feature_data = assembler.transform(purchase_data)
        
        # Split data
        (training, test) = feature_data.randomSplit([0.8, 0.2], seed=42)
        
        # Train Random Forest classifier
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="will_purchase",
            numTrees=100,
            seed=42
        )
        
        model = rf.fit(training)
        
        # Evaluate model
        predictions = model.transform(test)
        
        # Calculate accuracy
        correct_predictions = predictions.filter(
            col("prediction") == col("will_purchase")
        ).count()
        total_predictions = predictions.count()
        accuracy = correct_predictions / total_predictions
        
        self.logger.info(f"Purchase Prediction Accuracy: {accuracy}")
        
        return model, accuracy
    
    def create_hybrid_recommendations(self, cf_model, content_features, customer_index, 
                                    customer_history, num_recommendations=10, 
                                    cf_weight=0.7, content_weight=0.3):
        """
        Create hybrid recommendations combining collaborative and content-based filtering
        
        Args:
            cf_model: Trained collaborative filtering model
            content_features: Product content features
            customer_index: Customer index for CF
            customer_history: Customer's purchase history
            num_recommendations: Number of recommendations
            cf_weight: Weight for collaborative filtering
            content_weight: Weight for content-based filtering
            
        Returns:
            List of hybrid recommendations
        """
        # Get CF recommendations
        cf_recommendations = self.get_user_recommendations(cf_model, customer_index, num_recommendations * 2)
        cf_recs_pd = cf_recommendations.toPandas()
        
        # Get content-based recommendations based on purchase history
        content_recs = []
        for product_id in customer_history:
            similar_products = self.calculate_content_similarity(content_features, product_id)
            content_recs.extend(similar_products.to_dict('records'))
        
        # Combine and weight recommendations
        hybrid_scores = {}
        
        # Add CF scores
        for _, row in cf_recs_pd.iterrows():
            product_idx = row['product_index']
            if product_idx not in hybrid_scores:
                hybrid_scores[product_idx] = 0
            hybrid_scores[product_idx] += row['predicted_rating'] * cf_weight
        
        # Add content-based scores
        for item in content_recs:
            product_id = item['product_id']
            # Note: Need to map product_id to product_index here
            if product_id not in hybrid_scores:
                hybrid_scores[product_id] = 0
            hybrid_scores[product_id] += item['similarity_score'] * content_weight
        
        # Sort by combined score
        sorted_recommendations = sorted(hybrid_scores.items(), 
                                      key=lambda x: x[1], reverse=True)
        
        return sorted_recommendations[:num_recommendations]
    
    def save_models(self, models_dict, base_path):
        """
        Save trained models to disk
        
        Args:
            models_dict: Dictionary of models to save
            base_path: Base path for saving models
        """
        for model_name, model in models_dict.items():
            try:
                model_path = f"{base_path}/{model_name}"
                model.write().overwrite().save(model_path)
                self.logger.info(f"Saved {model_name} to {model_path}")
            except Exception as e:
                self.logger.error(f"Error saving {model_name}: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize models
    ml_models = RecommendationModels()
    
    # Example usage would go here
    # This would typically be called from the main pipeline