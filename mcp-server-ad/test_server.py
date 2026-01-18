"""
Test script for the Anomaly Detection MCP Server
Tests the server functionality with various scenarios.
"""

import json
import sys
import os

# Add parent directory to path to import from Data Generation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from server import detect_anomalies_core
import pandas as pd


def test_basic_functionality():
    """Test basic anomaly detection with simple data"""
    print("=" * 60)
    print("TEST 1: Basic Functionality")
    print("=" * 60)
    
    sample_data = [
        {"date": "2024-01-01", "sales": 100},
        {"date": "2024-01-02", "sales": 105},
        {"date": "2024-01-03", "sales": 98},
        {"date": "2024-01-04", "sales": 102},
        {"date": "2024-01-05", "sales": 500},  # Anomaly
        {"date": "2024-01-06", "sales": 103},
        {"date": "2024-01-07", "sales": 101},
    ]
    
    result = detect_anomalies_core(
        data=sample_data,  # Can pass list directly or JSON string
        time_column="date",
        value_column="sales",
        methods=["moving_average", "standard_deviation"],
        window=3,
        threshold=2.0
    )
    
    print(f"Total records: {result.get('total_records', 'N/A')}")
    print(f"Methods applied: {result.get('methods_applied', 'N/A')}")
    
    if 'results' in result:
        for method, data in result['results'].items():
            print(f"\n{method.upper()}:")
            print(f"  Total anomalies: {data.get('total_anomalies', 0)}")
            print(f"  Anomaly rate: {data.get('anomaly_rate', 0):.2%}")
    
    assert 'results' in result, "Results not found in response"
    assert 'moving_average' in result['results'], "Moving average method not found"
    assert 'standard_deviation' in result['results'], "Standard deviation method not found"
    print("✓ Test 1 PASSED\n")


def test_with_aggregation():
    """Test anomaly detection with aggregation level"""
    print("=" * 60)
    print("TEST 2: Aggregation Level")
    print("=" * 60)
    
    sample_data = [
        {"date": "2024-01-01", "product_id": "P001", "sales": 100},
        {"date": "2024-01-01", "product_id": "P002", "sales": 200},
        {"date": "2024-01-02", "product_id": "P001", "sales": 105},
        {"date": "2024-01-02", "product_id": "P002", "sales": 205},
        {"date": "2024-01-03", "product_id": "P001", "sales": 500},  # Anomaly
        {"date": "2024-01-03", "product_id": "P002", "sales": 198},
    ]
    
    result = detect_anomalies_core(
        data=sample_data,  # Can pass list directly or JSON string
        time_column="date",
        aggregation_level="product_id",
        value_column="sales",
        methods=["moving_average", "standard_deviation"],
        window=2,
        threshold=2.0
    )
    
    print(f"Aggregation level: {result.get('aggregation_level', 'N/A')}")
    print(f"Total records: {result.get('total_records', 'N/A')}")
    
    if 'results' in result:
        for method, data in result['results'].items():
            print(f"\n{method.upper()}:")
            print(f"  Total anomalies: {data.get('total_anomalies', 0)}")
    
    assert result.get('aggregation_level') == "product_id", "Aggregation level not set correctly"
    print("✓ Test 2 PASSED\n")


def test_all_methods():
    """Test all three detection methods"""
    print("=" * 60)
    print("TEST 3: All Methods (MA, SD, IQR)")
    print("=" * 60)
    
    sample_data = [
        {"date": "2024-01-01", "sales": 100},
        {"date": "2024-01-02", "sales": 105},
        {"date": "2024-01-03", "sales": 98},
        {"date": "2024-01-04", "sales": 102},
        {"date": "2024-01-05", "sales": 500},  # Anomaly
        {"date": "2024-01-06", "sales": 103},
        {"date": "2024-01-07", "sales": 101},
        {"date": "2024-01-08", "sales": 2},   # Anomaly
    ]
    
    result = detect_anomalies_core(
        data=sample_data,  # Can pass list directly or JSON string
        time_column="date",
        value_column="sales",
        methods=["moving_average", "standard_deviation", "iqr"],
        window=3,
        threshold=2.0,
        iqr_multiplier=1.5
    )
    
    print(f"Methods applied: {result.get('methods_applied', 'N/A')}")
    
    if 'results' in result:
        for method, data in result['results'].items():
            print(f"\n{method.upper()}:")
            print(f"  Total anomalies: {data.get('total_anomalies', 0)}")
            print(f"  Anomaly rate: {data.get('anomaly_rate', 0):.2%}")
    
    if 'combined_anomalies' in result:
        print(f"\nCOMBINED:")
        print(f"  Total anomalies: {result['combined_anomalies'].get('total_anomalies', 0)}")
        print(f"  Anomaly rate: {result['combined_anomalies'].get('anomaly_rate', 0):.2%}")
    
    assert len(result.get('methods_applied', [])) == 3, "Not all methods were applied"
    assert 'combined_anomalies' in result, "Combined anomalies not found"
    print("✓ Test 3 PASSED\n")


def test_auto_detect_value_column():
    """Test auto-detection of value column"""
    print("=" * 60)
    print("TEST 4: Auto-detect Value Column")
    print("=" * 60)
    
    sample_data = [
        {"date": "2024-01-01", "sales": 100, "price": 50.0},
        {"date": "2024-01-02", "sales": 105, "price": 50.0},
        {"date": "2024-01-03", "sales": 500, "price": 50.0},  # Anomaly
    ]
    
    result = detect_anomalies_core(
        data=sample_data,  # Can pass list directly or JSON string
        time_column="date",
        value_column=None,  # Auto-detect
        methods=["moving_average"],
        window=2,
        threshold=2.0
    )
    
    print(f"Auto-detected value column: {result.get('value_column', 'N/A')}")
    assert result.get('value_column') is not None, "Value column not auto-detected"
    print("✓ Test 4 PASSED\n")


def test_with_synthetic_data():
    """Test with synthetic data from the data generator"""
    print("=" * 60)
    print("TEST 5: Synthetic Data from CSV")
    print("=" * 60)
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data-gen', 'synthetic_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"⚠ CSV file not found at {csv_path}, skipping test")
        print("  Run 'python \"Data Generation/synthetic-data-gen.py\"' to generate test data")
        return
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from CSV")
        
        # Convert DataFrame to JSON
        data_json = df.to_json(orient='records', date_format='iso')
        
        result = detect_anomalies_core(
            data=df.to_dict('records'),  # Pass as list of dicts
            time_column="date",
            aggregation_level="product_id",
            value_column="sales",
            methods=["moving_average", "standard_deviation"],
            window=7,
            threshold=2.0
        )
        
        print(f"Total records: {result.get('total_records', 'N/A')}")
        if 'results' in result:
            for method, data in result['results'].items():
                print(f"{method.upper()}: {data.get('total_anomalies', 0)} anomalies")
        
        print("✓ Test 5 PASSED\n")
    except Exception as e:
        print(f"⚠ Error testing with CSV: {e}\n")


def test_error_handling():
    """Test error handling"""
    print("=" * 60)
    print("TEST 6: Error Handling")
    print("=" * 60)
    
    # Test invalid JSON (as string)
    result = detect_anomalies_core(
        data="invalid json",
        time_column="date",
        value_column="sales",
        methods=["moving_average"]
    )
    assert 'error' in result, "Error not detected for invalid JSON"
    print("✓ Invalid JSON handled correctly")
    
    # Test missing time column
    result = detect_anomalies_core(
        data=json.dumps([{"sales": 100}]),
        time_column="date",
        value_column="sales",
        methods=["moving_average"]
    )
    assert 'error' in result, "Error not detected for missing time column"
    print("✓ Missing time column handled correctly")
    
    # Test missing value column
    result = detect_anomalies_core(
        data=json.dumps([{"date": "2024-01-01"}]),
        time_column="date",
        value_column="sales",
        methods=["moving_average"]
    )
    assert 'error' in result, "Error not detected for missing value column"
    print("✓ Missing value column handled correctly")
    
    print("✓ Test 6 PASSED\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ANOMALY DETECTION MCP SERVER - TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_functionality,
        test_with_aggregation,
        test_all_methods,
        test_auto_detect_value_column,
        test_with_synthetic_data,
        test_error_handling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"✗ Test ERROR: {e}\n")
            failed += 1
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    print("=" * 60 + "\n")
    
    if failed == 0:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())

