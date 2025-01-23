import pandas as pd

# Load the datasets CSV files
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# Display the first few rows of each dataset
print(customers.head())
print(products.head())
print(transactions.head())

# Check for missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

# Check for duplicates
print(customers.duplicated().sum())
print(products.duplicated().sum())
print(transactions.duplicated().sum())

# Summary of each dataset
print(customers.info())
print(products.info())
print(transactions.info())

# Descriptive statistics
print(customers.describe())
print(products.describe())
print(transactions.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Customer distribution by region
region_counts = customers['Region'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=region_counts.index, y=region_counts.values)
plt.title("Customer Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Number of Customers")
plt.show()

# Product distribution by category
category_counts = products['Category'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title("Product Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Number of Products")
plt.show()

# Convert TransactionDate to datetime
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
transactions['Month'] = transactions['TransactionDate'].dt.to_period('M')

# Calculate monthly revenue
monthly_revenue = transactions.groupby('Month')['TotalValue'].sum()

# Plot monthly revenue
plt.figure(figsize=(10, 6))
monthly_revenue.index = monthly_revenue.index.astype(str)  # Convert Period to string for plotting
plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o', linestyle='-', color='blue')
plt.title("Monthly Revenue Trends")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.show()

# Top customers by revenue
top_customers = transactions.groupby('CustomerID')['TotalValue'].sum().sort_values(ascending=False).head(10)
print("Top 10 Customers by Revenue:")
print(top_customers)

# Top products by quantity sold
top_products = transactions.groupby('ProductID')['Quantity'].sum().sort_values(ascending=False).head(10)
print("Top 10 Products by Quantity Sold:")
print(top_products)

# Merge Transactions with Products
transactions_products = pd.merge(transactions, products, on='ProductID', how='left')

# Merge Transactions+Products with Customers
final_data = pd.merge(transactions_products, customers, on='CustomerID', how='left')

# Calculate total spending per customer
customer_summary = final_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',  # Total spending
    'TransactionID': 'count',  # Number of transactions
}).rename(columns={'TotalValue': 'TotalSpending', 'TransactionID': 'TransactionCount'})

# Reset index for easier access
customer_summary = customer_summary.reset_index()

# Add region to the summary
customer_features = pd.merge(customer_summary, customers[['CustomerID', 'Region']], on='CustomerID', how='left')

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Normalize the data
scaler = StandardScaler()
features = scaler.fit_transform(customer_features[['TotalSpending', 'TransactionCount']])

# Calculate similarity
similarity_matrix = cosine_similarity(features)


# Find top 3 similar customers for each customer
recommendations = {}
for idx, customer_id in enumerate(customer_features['CustomerID']):
    similar_indices = similarity_matrix[idx].argsort()[-4:-1][::-1]  # Top 3 excluding self
    similar_customers = [
        (customer_features.iloc[i]['CustomerID'], similarity_matrix[idx][i])
        for i in similar_indices
    ]
    recommendations[customer_id] = similar_customers

# Convert recommendations to DataFrame
recommendations_df = pd.DataFrame({
    'CustomerID': recommendations.keys(),
    'Lookalikes': [str(v) for v in recommendations.values()]
})

# Save to CSV
recommendations_df.to_csv('Lookalike.csv', index=False)

