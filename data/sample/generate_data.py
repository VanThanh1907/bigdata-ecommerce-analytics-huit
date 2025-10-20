"""
Generate sample e-commerce data for testing and demonstration
Author: HUIT Big Data Project
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import string
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class DataGenerator:
    def __init__(self):
        self.categories = [
            'Electronics', 'Clothing', 'Books', 'Home & Garden', 
            'Sports & Outdoors', 'Beauty & Health', 'Toys & Games',
            'Food & Beverages', 'Automotive', 'Jewelry'
        ]
        
        self.brands = [
            'Samsung', 'Apple', 'Nike', 'Adidas', 'Sony', 'LG', 'Canon',
            'Dell', 'HP', 'Asus', 'Zara', 'H&M', 'Uniqlo', 'Levi\'s'
        ]
        
        self.product_adjectives = [
            'Premium', 'Deluxe', 'Professional', 'Classic', 'Modern',
            'Smart', 'Wireless', 'Portable', 'Advanced', 'Eco-Friendly'
        ]
        
        self.product_nouns = [
            'Phone', 'Laptop', 'Tablet', 'Watch', 'Headphones', 'Camera',
            'Shirt', 'Jeans', 'Shoes', 'Bag', 'Book', 'Chair', 'Lamp'
        ]
        
        self.vietnamese_names = [
            'Nguyễn Văn An', 'Trần Thị Bình', 'Lê Văn Cường', 'Phạm Thị Dung',
            'Hoàng Văn Em', 'Vũ Thị Phương', 'Đặng Văn Giang', 'Ngô Thị Hoa',
            'Bùi Văn Inh', 'Lý Thị Kim', 'Phan Văn Long', 'Đinh Thị Mai',
            'Tạ Văn Nam', 'Chu Thị Oanh', 'Võ Văn Phú', 'Dương Thị Quỳnh'
        ]
    
    def generate_customer_id(self, index):
        """Generate customer ID"""
        return f"CUST{index:06d}"
    
    def generate_product_id(self, index):
        """Generate product ID"""
        return f"PROD{index:06d}"
    
    def generate_transaction_id(self, index):
        """Generate transaction ID"""
        return f"TXN{index:08d}"
    
    def generate_email(self, name):
        """Generate email from name"""
        # Convert Vietnamese name to simple format
        name_parts = name.lower().split()
        if len(name_parts) >= 2:
            username = name_parts[-1] + ''.join([n[0] for n in name_parts[:-1]])
        else:
            username = name_parts[0]
        
        # Remove Vietnamese characters (simplified)
        replacements = {
            'ă': 'a', 'â': 'a', 'á': 'a', 'à': 'a', 'ã': 'a', 'ạ': 'a',
            'ê': 'e', 'é': 'e', 'è': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ô': 'o', 'ơ': 'o', 'ó': 'o', 'ò': 'o', 'õ': 'o', 'ọ': 'o',
            'ư': 'u', 'ú': 'u', 'ù': 'u', 'ũ': 'u', 'ụ': 'u',
            'í': 'i', 'ì': 'i', 'ĩ': 'i', 'ị': 'i',
            'ý': 'y', 'ỳ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }
        
        for vn_char, en_char in replacements.items():
            username = username.replace(vn_char, en_char)
        
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        domain = random.choice(domains)
        
        return f"{username}{random.randint(1, 999)}@{domain}"
    
    def generate_customers(self, num_customers=1000):
        """Generate customer data"""
        customers = []
        
        for i in range(num_customers):
            customer_id = self.generate_customer_id(i + 1)
            name = random.choice(self.vietnamese_names)
            
            customer = {
                'customer_id': customer_id,
                'name': name,
                'email': self.generate_email(name),
                'phone': f"0{random.randint(100000000, 999999999)}",
                'age': random.randint(18, 65),
                'gender': random.choice(['Male', 'Female']),
                'city': random.choice([
                    'Ho Chi Minh City', 'Hanoi', 'Da Nang', 'Can Tho',
                    'Bien Hoa', 'Nha Trang', 'Hue', 'Vung Tau'
                ]),
                'registration_date': (
                    datetime.now() - timedelta(days=random.randint(1, 730))
                ).strftime('%Y-%m-%d'),
                'preferred_category': random.choice(self.categories)
            }
            customers.append(customer)
        
        return pd.DataFrame(customers)
    
    def generate_products(self, num_products=500):
        """Generate product data"""
        products = []
        
        for i in range(num_products):
            product_id = self.generate_product_id(i + 1)
            category = random.choice(self.categories)
            brand = random.choice(self.brands)
            
            # Generate product name
            adj = random.choice(self.product_adjectives)
            noun = random.choice(self.product_nouns)
            product_name = f"{brand} {adj} {noun}"
            
            # Generate price based on category
            base_prices = {
                'Electronics': (500000, 50000000),
                'Clothing': (100000, 5000000),
                'Books': (50000, 500000),
                'Home & Garden': (200000, 10000000),
                'Sports & Outdoors': (300000, 15000000),
                'Beauty & Health': (100000, 2000000),
                'Toys & Games': (100000, 3000000),
                'Food & Beverages': (20000, 500000),
                'Automotive': (1000000, 100000000),
                'Jewelry': (500000, 50000000)
            }
            
            min_price, max_price = base_prices.get(category, (100000, 5000000))
            price = random.randint(min_price, max_price)
            
            product = {
                'product_id': product_id,
                'product_name': product_name,
                'category': category,
                'brand': brand,
                'price': price,
                'description': f"High quality {product_name.lower()} from {brand}",
                'rating': round(random.uniform(3.0, 5.0), 1),
                'stock_quantity': random.randint(0, 1000),
                'weight': round(random.uniform(0.1, 50.0), 2),
                'dimensions': f"{random.randint(5, 50)}x{random.randint(5, 50)}x{random.randint(5, 50)} cm"
            }
            products.append(product)
        
        return pd.DataFrame(products)
    
    def generate_transactions(self, customers_df, products_df, num_transactions=10000):
        """Generate transaction data"""
        transactions = []
        
        # Create weighted selection for active customers (80/20 rule)
        customer_ids = customers_df['customer_id'].tolist()
        customer_weights = [0.8 if i < len(customer_ids) * 0.2 else 0.2 
                          for i in range(len(customer_ids))]
        
        for i in range(num_transactions):
            transaction_id = self.generate_transaction_id(i + 1)
            
            # Select customer (weighted towards active customers)
            customer_id = np.random.choice(customer_ids, p=np.array(customer_weights)/sum(customer_weights))
            
            # Get customer info for preferences
            customer_info = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
            preferred_category = customer_info['preferred_category']
            
            # Select product (70% chance from preferred category)
            if random.random() < 0.7:
                category_products = products_df[products_df['category'] == preferred_category]
                if not category_products.empty:
                    product = category_products.sample(1).iloc[0]
                else:
                    product = products_df.sample(1).iloc[0]
            else:
                product = products_df.sample(1).iloc[0]
            
            # Generate transaction details
            quantity = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3])[0]
            
            # Add some price variation (discounts/promotions)
            price_variation = random.uniform(0.8, 1.0)
            final_price = int(product['price'] * price_variation)
            
            # Generate timestamp (more recent transactions are more likely)
            days_ago = int(np.random.exponential(30))  # Exponential distribution
            days_ago = min(days_ago, 365)  # Cap at 1 year
            
            timestamp = datetime.now() - timedelta(days=days_ago, 
                                                 hours=random.randint(0, 23),
                                                 minutes=random.randint(0, 59))
            
            transaction = {
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'product_id': product['product_id'],
                'category': product['category'],
                'quantity': quantity,
                'price': final_price,
                'total_amount': final_price * quantity,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'payment_method': random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery']),
                'shipping_cost': random.choice([0, 30000, 50000, 100000]),
                'discount_amount': random.randint(0, final_price // 10)
            }
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def save_data(self, customers_df, products_df, transactions_df, output_dir="data/sample"):
        """Save generated data to files"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CSV files
        customers_df.to_csv(f"{output_dir}/customers.csv", index=False, encoding='utf-8-sig')
        products_df.to_csv(f"{output_dir}/products.csv", index=False, encoding='utf-8-sig')
        transactions_df.to_csv(f"{output_dir}/transactions.csv", index=False, encoding='utf-8-sig')
        
        # Save JSON files for MongoDB import
        customers_df.to_json(f"{output_dir}/customers.json", orient='records', lines=True)
        products_df.to_json(f"{output_dir}/products.json", orient='records', lines=True)
        transactions_df.to_json(f"{output_dir}/transactions.json", orient='records', lines=True)
        
        # Generate statistics
        stats = {
            'generation_date': datetime.now().isoformat(),
            'total_customers': len(customers_df),
            'total_products': len(products_df),
            'total_transactions': len(transactions_df),
            'date_range': {
                'start': transactions_df['timestamp'].min(),
                'end': transactions_df['timestamp'].max()
            },
            'categories': list(products_df['category'].unique()),
            'total_revenue': float(transactions_df['total_amount'].sum())
        }
        
        with open(f"{output_dir}/data_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Data generated successfully!")
        print(f"- Customers: {len(customers_df):,}")
        print(f"- Products: {len(products_df):,}")
        print(f"- Transactions: {len(transactions_df):,}")
        print(f"- Total Revenue: {transactions_df['total_amount'].sum():,.0f} VND")
        print(f"- Files saved to: {output_dir}/")

def main():
    """Generate sample data"""
    generator = DataGenerator()
    
    print("Generating sample e-commerce data...")
    
    # Generate data
    customers_df = generator.generate_customers(1000)
    products_df = generator.generate_products(500)
    transactions_df = generator.generate_transactions(customers_df, products_df, 10000)
    
    # Save data
    generator.save_data(customers_df, products_df, transactions_df)

if __name__ == "__main__":
    main()