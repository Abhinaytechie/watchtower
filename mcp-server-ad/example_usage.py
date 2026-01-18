"""
Example usage of the Anomaly Detection MCP Server
This demonstrates how to use the server with sample data.
"""

import json
from server import detect_anomalies_core

# Sample time series data
sample_data = [
    {"date": "2024-01-01", "product_id": "P001", "sales": 100, "price": 50.0},
    {"date": "2024-01-02", "product_id": "P001", "sales": 105, "price": 50.0},
    {"date": "2024-01-03", "product_id": "P001", "sales": 98, "price": 50.0},
    {"date": "2024-01-04", "product_id": "P001", "sales": 102, "price": 50.0},
    {"date": "2024-01-05", "product_id": "P001", "sales": 500, "price": 50.0},  # Anomaly
    {"date": "2024-01-06", "product_id": "P001", "sales": 103, "price": 50.0},
    {"date": "2024-01-07", "product_id": "P001", "sales": 101, "price": 50.0},
    {"date": "2024-01-08", "product_id": "P001", "sales": 99, "price": 50.0},
    {"date": "2024-01-09", "product_id": "P001", "sales": 105, "price": 50.0},
    {"date": "2024-01-10", "product_id": "P001", "sales": 2, "price": 50.0},  # Anomaly
]

if __name__ == "__main__":
    # Convert data to JSON string
    data_json = json.dumps(sample_data)
    
    # Example 1: Using moving average and standard deviation methods
    print("Example 1: Moving Average and Standard Deviation")
    result = detect_anomalies_core(
        data=data_json,
        time_column="date",
        value_column="sales",
        methods=["moving_average", "standard_deviation"],
        window=3,
        threshold=2.0
    )
    print(json.dumps(result, indent=2))
    print("\n" + "="*50 + "\n")
    
    # Example 2: With aggregation level
    sample_data_with_products = [
        {"date": "2024-01-01", "product_id": "P001", "sales": 100},
        {"date": "2024-01-01", "product_id": "P002", "sales": 200},
        {"date": "2024-01-02", "product_id": "P001", "sales": 105},
        {"date": "2024-01-02", "product_id": "P002", "sales": 205},
        {"date": "2024-01-03", "product_id": "P001", "sales": 500},  # Anomaly
        {"date": "2024-01-03", "product_id": "P002", "sales": 198},
    ]
    
    print("Example 2: With Aggregation Level")
    result2 = detect_anomalies_core(
        data=json.dumps(sample_data_with_products),
        time_column="date",
        aggregation_level="product_id",
        value_column="sales",
        methods=["moving_average", "standard_deviation", "iqr"],
        window=2,
        threshold=2.0
    )
    print(json.dumps(result2, indent=2))

