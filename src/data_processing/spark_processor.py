"""
Spark Data Processing Pipeline for E-commerce Analytics
Author: HUIT Big Data Project
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
from datetime import datetime
import logging

class EcommerceDataProcessor:
    def __init__(self, app_name="EcommerceAnalytics"):
        """Initialize Spark session and configurations"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.logger = logging.getLogger(__name__)
        
    def load_raw_data(self, file_path, file_type="csv"):
        """
        Load raw e-commerce data from various sources
        
        Args:
            file_path (str): Path to data file
            file_type (str): Type of file (csv, json, parquet)
        
        Returns:
            DataFrame: Spark DataFrame containing raw data
        """
        try:
            if file_type.lower() == "csv":
                df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            elif file_type.lower() == "json":
                df = self.spark.read.json(file_path)
            elif file_type.lower() == "parquet":
                df = self.spark.read.parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            self.logger.info(f"Successfully loaded {df.count()} records from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def clean_customer_data(self, customers_df):
        """
        Clean and preprocess customer data
        
        Args:
            customers_df: Raw customers DataFrame
            
        Returns:
            DataFrame: Cleaned customers data
        """
        # Remove duplicates and null values
        cleaned_df = customers_df.dropDuplicates(["customer_id"]) \
                                .filter(col("customer_id").isNotNull())
        
        # Standardize email format
        cleaned_df = cleaned_df.withColumn("email", 
                                         lower(trim(col("email"))))
        
        # Extract age groups
        cleaned_df = cleaned_df.withColumn("age_group", 
                                         when(col("age") < 25, "18-24")
                                         .when(col("age") < 35, "25-34")
                                         .when(col("age") < 45, "35-44")
                                         .when(col("age") < 55, "45-54")
                                         .otherwise("55+"))
        
        # Clean phone numbers (remove non-digits)
        cleaned_df = cleaned_df.withColumn("phone", 
                                         regexp_replace(col("phone"), "[^0-9]", ""))
        
        return cleaned_df
    
    def clean_transaction_data(self, transactions_df):
        """
        Clean and preprocess transaction data
        
        Args:
            transactions_df: Raw transactions DataFrame
            
        Returns:
            DataFrame: Cleaned transactions data
        """
        # Filter valid transactions
        cleaned_df = transactions_df.filter(
            (col("transaction_id").isNotNull()) &
            (col("customer_id").isNotNull()) &
            (col("product_id").isNotNull()) &
            (col("quantity") > 0) &
            (col("price") > 0)
        )
        
        # Calculate total amount
        cleaned_df = cleaned_df.withColumn("total_amount", 
                                         col("quantity") * col("price"))
        
        # Add time features
        cleaned_df = cleaned_df.withColumn("transaction_date", 
                                         to_date(col("timestamp")))
        
        cleaned_df = cleaned_df.withColumn("hour", 
                                         hour(col("timestamp")))
        
        cleaned_df = cleaned_df.withColumn("day_of_week", 
                                         dayofweek(col("timestamp")))
        
        cleaned_df = cleaned_df.withColumn("month", 
                                         month(col("timestamp")))
        
        # Add day part classification
        cleaned_df = cleaned_df.withColumn("day_part",
                                         when(col("hour") < 6, "Night")
                                         .when(col("hour") < 12, "Morning")
                                         .when(col("hour") < 18, "Afternoon")
                                         .otherwise("Evening"))
        
        return cleaned_df
    
    def clean_product_data(self, products_df):
        """
        Clean and preprocess product data
        
        Args:
            products_df: Raw products DataFrame
            
        Returns:
            DataFrame: Cleaned products data
        """
        # Remove duplicates and null product IDs
        cleaned_df = products_df.dropDuplicates(["product_id"]) \
                               .filter(col("product_id").isNotNull())
        
        # Clean product names
        cleaned_df = cleaned_df.withColumn("product_name", 
                                         trim(col("product_name")))
        
        # Standardize categories
        cleaned_df = cleaned_df.withColumn("category", 
                                         lower(trim(col("category"))))
        
        # Add price ranges
        cleaned_df = cleaned_df.withColumn("price_range",
                                         when(col("price") < 100000, "Budget")
                                         .when(col("price") < 500000, "Mid-range")
                                         .when(col("price") < 1000000, "Premium")
                                         .otherwise("Luxury"))
        
        # Clean brand names
        cleaned_df = cleaned_df.withColumn("brand", 
                                         when(col("brand").isNull(), "Unknown")
                                         .otherwise(trim(col("brand"))))
        
        return cleaned_df
    
    def create_customer_features(self, transactions_df, customers_df):
        """
        Create customer behavior features for ML models
        
        Args:
            transactions_df: Cleaned transactions data
            customers_df: Cleaned customers data
            
        Returns:
            DataFrame: Customer features for ML
        """
        # Aggregate customer transaction behavior
        customer_agg = transactions_df.groupBy("customer_id").agg(
            count("transaction_id").alias("total_transactions"),
            sum("total_amount").alias("total_spent"),
            avg("total_amount").alias("avg_order_value"),
            max("transaction_date").alias("last_purchase_date"),
            min("transaction_date").alias("first_purchase_date"),
            countDistinct("product_id").alias("unique_products"),
            countDistinct("category").alias("unique_categories")
        )
        
        # Calculate recency, frequency, monetary (RFM) features
        current_date = lit(datetime.now().date())
        
        customer_agg = customer_agg.withColumn("recency_days",
                                             datediff(current_date, col("last_purchase_date")))
        
        customer_agg = customer_agg.withColumn("customer_lifetime_days",
                                             datediff(col("last_purchase_date"), col("first_purchase_date")))
        
        # Join with customer demographic data
        customer_features = customers_df.join(customer_agg, "customer_id", "left")
        
        # Fill null values for new customers
        customer_features = customer_features.fillna(0, ["total_transactions", "total_spent", 
                                                       "avg_order_value", "unique_products", 
                                                       "unique_categories"])
        
        # Add customer segments based on RFM
        customer_features = customer_features.withColumn("customer_segment",
            when((col("recency_days") <= 30) & (col("total_transactions") >= 5) & 
                 (col("total_spent") >= 1000000), "Champions")
            .when((col("recency_days") <= 60) & (col("total_transactions") >= 3), "Loyal Customers")
            .when((col("recency_days") <= 90) & (col("total_spent") >= 500000), "Potential Loyalists")
            .when(col("recency_days") <= 180, "New Customers")
            .when((col("recency_days") > 180) & (col("total_transactions") >= 3), "At Risk")
            .when(col("recency_days") > 365, "Hibernating")
            .otherwise("Lost Customers")
        )
        
        return customer_features
    
    def create_product_features(self, transactions_df, products_df):
        """
        Create product performance features
        
        Args:
            transactions_df: Cleaned transactions data
            products_df: Cleaned products data
            
        Returns:
            DataFrame: Product features with performance metrics
        """
        # Aggregate product performance
        product_agg = transactions_df.groupBy("product_id").agg(
            count("transaction_id").alias("total_sales"),
            sum("quantity").alias("total_quantity_sold"),
            sum("total_amount").alias("total_revenue"),
            avg("total_amount").alias("avg_sale_amount"),
            countDistinct("customer_id").alias("unique_customers"),
            max("transaction_date").alias("last_sale_date")
        )
        
        # Calculate product popularity metrics
        total_customers = transactions_df.select("customer_id").distinct().count()
        
        product_agg = product_agg.withColumn("customer_penetration",
                                           col("unique_customers") / total_customers)
        
        # Join with product information
        product_features = products_df.join(product_agg, "product_id", "left")
        
        # Fill null values for products with no sales
        product_features = product_features.fillna(0, ["total_sales", "total_quantity_sold",
                                                     "total_revenue", "avg_sale_amount",
                                                     "unique_customers", "customer_penetration"])
        
        # Add product performance categories
        product_features = product_features.withColumn("performance_category",
            when(col("total_sales") >= 100, "Best Seller")
            .when(col("total_sales") >= 50, "Popular")
            .when(col("total_sales") >= 10, "Average")
            .when(col("total_sales") > 0, "Slow Moving")
            .otherwise("No Sales")
        )
        
        return product_features
    
    def save_processed_data(self, df, output_path, format="parquet", mode="overwrite"):
        """
        Save processed data to storage
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            format: Output format (parquet, csv, json)
            mode: Save mode (overwrite, append)
        """
        try:
            if format.lower() == "parquet":
                df.write.mode(mode).parquet(output_path)
            elif format.lower() == "csv":
                df.write.mode(mode).option("header", "true").csv(output_path)
            elif format.lower() == "json":
                df.write.mode(mode).json(output_path)
            
            self.logger.info(f"Successfully saved data to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving data to {output_path}: {str(e)}")
            raise
    
    def run_full_pipeline(self, raw_data_paths, output_base_path):
        """
        Run the complete data processing pipeline
        
        Args:
            raw_data_paths: Dictionary with paths to raw data files
            output_base_path: Base path for processed data output
        """
        try:
            # Load raw data
            customers_raw = self.load_raw_data(raw_data_paths['customers'])
            transactions_raw = self.load_raw_data(raw_data_paths['transactions'])
            products_raw = self.load_raw_data(raw_data_paths['products'])
            
            # Clean data
            customers_clean = self.clean_customer_data(customers_raw)
            transactions_clean = self.clean_transaction_data(transactions_raw)
            products_clean = self.clean_product_data(products_raw)
            
            # Create features
            customer_features = self.create_customer_features(transactions_clean, customers_clean)
            product_features = self.create_product_features(transactions_clean, products_clean)
            
            # Save processed data
            self.save_processed_data(customers_clean, f"{output_base_path}/customers_clean")
            self.save_processed_data(transactions_clean, f"{output_base_path}/transactions_clean")
            self.save_processed_data(products_clean, f"{output_base_path}/products_clean")
            self.save_processed_data(customer_features, f"{output_base_path}/customer_features")
            self.save_processed_data(product_features, f"{output_base_path}/product_features")
            
            self.logger.info("Data processing pipeline completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            self.spark.stop()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize processor
    processor = EcommerceDataProcessor()
    
    # Define data paths
    raw_data_paths = {
        'customers': '/data/raw/customers.csv',
        'transactions': '/data/raw/transactions.csv',
        'products': '/data/raw/products.csv'
    }
    
    # Run pipeline
    processor.run_full_pipeline(raw_data_paths, '/data/processed')