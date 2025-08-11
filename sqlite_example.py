import sqlite3
import pandas as pd

# Step 1: Connect to (or create) SQLite database and create tables
conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT,
    order_status TEXT,
    order_purchase_timestamp TEXT,
    order_approved_at TEXT,
    order_delivered_carrier_date TEXT,
    order_delivered_customer_date TEXT,
    order_estimated_delivery_date TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS order_items (
    order_id TEXT,
    order_item_id INTEGER,
    product_id TEXT,
    seller_id TEXT,
    shipping_limit_date TEXT,
    price REAL,
    freight_value REAL
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    customer_unique_id TEXT,
    customer_zip_code_prefix INTEGER,
    customer_city TEXT,
    customer_state TEXT
);
''')

conn.commit()

# Step 2: Check if tables are empty, if yes load CSVs and insert data
def is_table_empty(table_name):
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    return count == 0

if is_table_empty('orders') and is_table_empty('order_items') and is_table_empty('customers'):
    print("Loading CSV files into SQLite tables...")
    orders_df = pd.read_csv("C:/Users/SNEHA SAHA/OneDrive/Desktop/E-commerce_Funnel_and_Customer_Lifetime_Value_(CLV)/data/olist/olist_orders_dataset.csv")
    order_items_df = pd.read_csv("C:/Users/SNEHA SAHA/OneDrive/Desktop/E-commerce_Funnel_and_Customer_Lifetime_Value_(CLV)/data/olist/olist_order_items_dataset.csv")
    customers_df = pd.read_csv("C:/Users/SNEHA SAHA/OneDrive/Desktop/E-commerce_Funnel_and_Customer_Lifetime_Value_(CLV)/data/olist/olist_customers_dataset.csv")

    orders_df.to_sql('orders', conn, if_exists='replace', index=False)
    order_items_df.to_sql('order_items', conn, if_exists='replace', index=False)
    customers_df.to_sql('customers', conn, if_exists='replace', index=False)
    print("CSV data loaded into SQLite tables.")
else:
    print("Data already exists in tables, skipping CSV loading.")

# Step 3: Query joined data filtered by delivered orders
query = """
SELECT o.order_id, o.customer_id, o.order_status, o.order_purchase_timestamp,
       o.order_approved_at, o.order_delivered_carrier_date, o.order_delivered_customer_date,
       o.order_estimated_delivery_date,
       oi.order_item_id, oi.product_id, oi.seller_id, oi.shipping_limit_date,
       oi.price, oi.freight_value,
       c.customer_unique_id, c.customer_zip_code_prefix, c.customer_city, c.customer_state
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status = 'delivered'
"""

df = pd.read_sql_query(query, conn)

print("\nSample of joined delivered orders data:")
print(df.head())


import numpy as np

# Convert order_purchase_timestamp to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Calculate total revenue per order (sum price per order_id)
order_revenue = df.groupby('order_id')['price'].sum().reset_index(name='order_revenue')

# Calculate revenue per customer (sum of order revenues)
customer_revenue = df.groupby('customer_id')['price'].sum().reset_index(name='total_revenue')

# Calculate purchase frequency per customer (number of distinct orders)
purchase_frequency = df.groupby('customer_id')['order_id'].nunique().reset_index(name='order_count')

# Calculate active duration (days between first and last purchase) per customer
customer_dates = df.groupby('customer_id')['order_purchase_timestamp'].agg(['min', 'max']).reset_index()
customer_dates['active_days'] = (customer_dates['max'] - customer_dates['min']).dt.days
customer_dates['active_years'] = customer_dates['active_days'] / 365
customer_dates['active_years'] = customer_dates['active_years'].replace(0, 1/365)  # avoid zero division

# Merge these dataframes to get one CLV dataframe
clv_df = customer_revenue.merge(purchase_frequency, on='customer_id')
clv_df = clv_df.merge(customer_dates[['customer_id', 'active_years']], on='customer_id')

# Calculate Average Order Value (AOV)
clv_df['avg_order_value'] = clv_df['total_revenue'] / clv_df['order_count']

# Calculate purchase frequency per year
clv_df['purchase_frequency_per_year'] = clv_df['order_count'] / clv_df['active_years']

# Assume average customer lifespan (years) and gross margin
customer_lifespan_years = 3
gross_margin = 0.6

# Calculate CLV
clv_df['CLV'] = clv_df['avg_order_value'] * clv_df['purchase_frequency_per_year'] * customer_lifespan_years * gross_margin

print("\nSample CLV calculations:")
print(clv_df.head())

# Save CLV dataframe to CSV for future use
clv_df.to_csv("customer_clv.csv", index=False)
print("\nCLV data saved to customer_clv.csv")

import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# Plot histogram of CLV
plt.figure(figsize=(10,6))
sns.histplot(clv_df['CLV'], bins=50, kde=True)
plt.title('Distribution of Customer Lifetime Value (CLV)')
plt.xlabel('CLV')
plt.ylabel('Number of Customers')
plt.show()

# Plot boxplot to check for outliers
plt.figure(figsize=(8,4))
sns.boxplot(x=clv_df['CLV'])
plt.title('Boxplot of Customer Lifetime Value (CLV)')
plt.xlabel('CLV')
plt.show()

# Segment customers into Low, Medium, High CLV based on quantiles
quantiles = clv_df['CLV'].quantile([0.33, 0.66]).values

def clv_segment(clv):
    if clv <= quantiles[0]:
        return 'Low CLV'
    elif clv <= quantiles[1]:
        return 'Medium CLV'
    else:
        return 'High CLV'

clv_df['CLV_Segment'] = clv_df['CLV'].apply(clv_segment)

# Show counts of each segment
print("\nCustomer counts by CLV Segment:")
print(clv_df['CLV_Segment'].value_counts())

# Visualize CLV by segment
plt.figure(figsize=(10,6))
sns.boxplot(x='CLV_Segment', y='CLV', data=clv_df, order=['Low CLV', 'Medium CLV', 'High CLV'])
plt.title('CLV Distribution by Customer Segment')
plt.xlabel('CLV Segment')
plt.ylabel('Customer Lifetime Value')
plt.show()

import sqlite3
import pandas as pd

# Connect to SQLite DB
conn = sqlite3.connect('my_database.db')

# Extract data
orders = pd.read_sql_query("SELECT * FROM orders", conn)
order_items = pd.read_sql_query("SELECT * FROM order_items", conn)
customers = pd.read_sql_query("SELECT * FROM customers", conn)

conn.close()

# Data Cleaning

# 1. Convert date columns to datetime
date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_cols:
    orders[col] = pd.to_datetime(orders[col], errors='coerce')  # coerce invalid dates to NaT

# 2. Check and handle missing values
print("Missing values in orders:")
print(orders.isnull().sum())

# For example, drop rows with missing essential dates (like purchase timestamp)
orders = orders.dropna(subset=['order_purchase_timestamp'])

# 3. Remove duplicates if any
orders = orders.drop_duplicates()
order_items = order_items.drop_duplicates()
customers = customers.drop_duplicates()

# 4. Filter orders with valid statuses (e.g., delivered or shipped only)
valid_statuses = ['delivered', 'shipped']
orders = orders[orders['order_status'].isin(valid_statuses)]

# Now your data is cleaned and ready for analysis!

# You can save the cleaned data back to CSV or database if needed
orders.to_csv('cleaned_orders.csv', index=False)
order_items.to_csv('cleaned_order_items.csv', index=False)
customers.to_csv('cleaned_customers.csv', index=False)



conn.close()
