import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("\n--- Loading Datasets ---")
orders = pd.read_csv("C:/Users/SNEHA SAHA/OneDrive/Desktop/E-commerce_Funnel_and_Customer_Lifetime_Value_(CLV)/data/olist/olist_orders_dataset.csv")
print("Orders columns:", orders.columns)

order_items = pd.read_csv("C:/Users/SNEHA SAHA/OneDrive/Desktop/E-commerce_Funnel_and_Customer_Lifetime_Value_(CLV)/data/olist/olist_order_items_dataset.csv")
print("Order Items columns:", order_items.columns)

customers = pd.read_csv("C:/Users/SNEHA SAHA/OneDrive/Desktop/E-commerce_Funnel_and_Customer_Lifetime_Value_(CLV)/data/olist/olist_customers_dataset.csv")
print("Customers columns:", customers.columns)

print(f"Orders shape: {orders.shape}")
print(f"Order Items shape: {order_items.shape}")
print(f"Customers shape: {customers.shape}")

# Merge datasets: orders + order_items + customers
print("\n--- Merging datasets ---")
df = orders.merge(order_items, on="order_id")
df = df.merge(customers, on="customer_id")

print(f"Combined dataset shape: {df.shape}")

# Convert dates to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

# Calculate active duration per customer
orders_dates = orders.groupby('customer_id')['order_purchase_timestamp'].agg(['min', 'max']).reset_index()
orders_dates['active_days'] = (orders_dates['max'] - orders_dates['min']).dt.days
orders_dates['active_years'] = orders_dates['active_days'] / 365
orders_dates['active_years'] = orders_dates['active_years'].replace(0, 1/365)  # avoid division by zero

# Step 1: Funnel Metrics
print("\n--- Funnel Metrics ---")
total_customers = customers['customer_id'].nunique()
total_orders = orders['order_id'].nunique()
delivered_orders = orders[orders['order_status'] == 'delivered']['order_id'].nunique()

print(f"Total Customers: {total_customers}")
print(f"Total Orders Placed: {total_orders}")
print(f"Total Orders Delivered: {delivered_orders}")

# Calculate funnel conversion rates
order_rate = total_orders / total_customers * 100
delivery_rate = delivered_orders / total_orders * 100

print(f"Customer to Order Conversion Rate: {order_rate:.2f}%")
print(f"Order to Delivery Conversion Rate: {delivery_rate:.2f}%")

# Step 2: Calculate Customer Lifetime Value (CLV)
print("\n--- Calculating CLV ---")

# Calculate total revenue per order (sum price per order_id)
order_revenue = df.groupby('order_id')['price'].sum().reset_index(name='order_revenue')

# Merge order revenue back to orders + customers
orders_revenue = orders.merge(order_revenue, on='order_id')

# Calculate revenue per customer
customer_revenue = orders_revenue.groupby('customer_id')['order_revenue'].sum().reset_index(name='total_revenue')

# Calculate purchase frequency per customer (number of orders)
purchase_frequency = orders.groupby('customer_id')['order_id'].count().reset_index(name='order_count')

# Merge revenue and frequency
clv_df = customer_revenue.merge(purchase_frequency, on='customer_id')

# Average Order Value (AOV)
clv_df['avg_order_value'] = clv_df['total_revenue'] / clv_df['order_count']

# Assume average customer lifespan (years) and gross margin
customer_lifespan_years = 3
gross_margin = 0.6

# Merge active years info
clv_df = clv_df.merge(orders_dates[['customer_id', 'active_years']], on='customer_id')

# Purchase frequency per year (orders per customer per year)
clv_df['purchase_frequency_per_year'] = clv_df['order_count'] / clv_df['active_years']

# Calculate CLV per customer
clv_df['CLV'] = clv_df['avg_order_value'] * clv_df['purchase_frequency_per_year'] * customer_lifespan_years * gross_margin

# Overall average CLV
avg_clv = clv_df['CLV'].mean()

print(f"Average Customer Lifetime Value (CLV): ${avg_clv:.2f}")

# Optional: show top 5 customers by CLV
print("\nTop 5 Customers by CLV:")
print(clv_df[['customer_id', 'CLV']].sort_values(by='CLV', ascending=False).head())

# Save CLV data to CSV
clv_df.to_csv("customer_clv.csv", index=False)
print("CLV data saved to customer_clv.csv")

# --- Step 3: Visualize CLV distribution ---

sns.set(style="whitegrid")

plt.figure(figsize=(10,6))
sns.histplot(clv_df['CLV'], bins=50, kde=True)
plt.title('Distribution of Customer Lifetime Value (CLV)')
plt.xlabel('CLV')
plt.ylabel('Number of Customers')
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x=clv_df['CLV'])
plt.title('Boxplot of Customer Lifetime Value (CLV)')
plt.xlabel('CLV')
plt.show()

# --- Step 4: Analyze factors influencing CLV ---

clv_analysis_df = clv_df.merge(customers, on='customer_id')

avg_clv_by_state = clv_analysis_df.groupby('customer_state')['CLV'].mean().sort_values(ascending=False)
print("\nAverage CLV by Customer State:")
print(avg_clv_by_state)

plt.figure(figsize=(12,6))
sns.barplot(x=avg_clv_by_state.index, y=avg_clv_by_state.values, palette='viridis')
plt.title('Average Customer Lifetime Value by State')
plt.xlabel('State')
plt.ylabel('Average CLV')
plt.xticks(rotation=45)
plt.show()

numeric_cols = clv_df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns in CLV data:", numeric_cols)

corr = clv_df[numeric_cols].corr()
print("\nCorrelation matrix:")
print(corr['CLV'].sort_values(ascending=False))

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# --- Step 5: Segment Customers by CLV ---

quantiles = clv_df['CLV'].quantile([0.33, 0.66]).values

def clv_segment(clv):
    if clv <= quantiles[0]:
        return 'Low CLV'
    elif clv <= quantiles[1]:
        return 'Medium CLV'
    else:
        return 'High CLV'

clv_df['CLV_Segment'] = clv_df['CLV'].apply(clv_segment)

print("\nCustomer counts by CLV Segment:")
print(clv_df['CLV_Segment'].value_counts())

segmented_customers = clv_df.merge(customers, on='customer_id')

plt.figure(figsize=(10,6))
sns.boxplot(x='CLV_Segment', y='CLV', data=segmented_customers, order=['Low CLV', 'Medium CLV', 'High CLV'])
plt.title('CLV Distribution by Customer Segment')
plt.xlabel('CLV Segment')
plt.ylabel('Customer Lifetime Value')
plt.show()

# --- Step 6: Analyze Customer Behavior by CLV Segment ---

segment_summary = clv_df.groupby('CLV_Segment').agg({
    'order_count': 'mean',
    'avg_order_value': 'mean',
    'active_years': 'mean',
    'CLV': ['mean', 'count']
}).round(2)

segment_summary.columns = ['Avg Order Count', 'Avg Order Value', 'Avg Active Years', 'Avg CLV', 'Customer Count']

print("\nCustomer Behavior Summary by CLV Segment:")
print(segment_summary)

plt.figure(figsize=(8,5))
sns.barplot(x=segment_summary.index, y='Avg Order Count', data=segment_summary)
plt.title('Average Order Count by CLV Segment')
plt.ylabel('Average Order Count')
plt.xlabel('CLV Segment')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x=segment_summary.index, y='Avg Order Value', data=segment_summary)
plt.title('Average Order Value by CLV Segment')
plt.ylabel('Average Order Value')
plt.xlabel('CLV Segment')
plt.show()

# --- Step 7: Classification Model to Predict CLV Segment ---

features = clv_df[['order_count', 'avg_order_value', 'active_years', 'purchase_frequency_per_year']]
target = clv_df['CLV_Segment']

le = LabelEncoder()
target_encoded = le.fit_transform(target)

X_train, X_test, y_train, y_test = train_test_split(features, target_encoded, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix for CLV Segment Classification')
plt.show()

# --- Step 8: Regression Model to Predict CLV Value ---

train, test = train_test_split(clv_df, test_size=0.3, random_state=42)

features_reg = ['total_revenue', 'order_count', 'avg_order_value', 'active_years', 'purchase_frequency_per_year']
X_train_reg = train[features_reg]
y_train_reg = train['CLV']
X_test_reg = test[features_reg]
y_test_reg = test['CLV']

model = LinearRegression()
model.fit(X_train_reg, y_train_reg)

y_pred_reg = model.predict(X_test_reg)

mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Test MSE: {mse:.2f}")
print(f"Test R2 Score: {r2:.2f}")

joblib.dump(model, "clv_model.pkl")
print("Model saved to clv_model.pkl")
