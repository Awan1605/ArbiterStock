"""
API Endpoints Testing Script
Complete testing for all API endpoints
"""

import requests
import json
from colorama import init, Fore, Style
import sys

init()  # Initialize colorama

BASE_URL = "http://localhost:8000"

def print_header(text):
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Style.RESET_ALL}\n")

def print_success(text):
    print(f"{Fore.GREEN}✓ {text}{Style.RESET_ALL}")

def print_error(text):
    print(f"{Fore.RED}✗ {text}{Style.RESET_ALL}")

def print_info(text):
    print(f"{Fore.YELLOW}→ {text}{Style.RESET_ALL}")

def test_endpoint(method, endpoint, description):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print_info(f"Testing: {description}")
    print_info(f"URL: {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Status: {response.status_code}")
            
            # Show preview of response
            json_str = json.dumps(data, indent=2)
            preview = json_str[:300] + "..." if len(json_str) > 300 else json_str
            print(f"{Fore.WHITE}{preview}{Style.RESET_ALL}")
            
            return True, data
        else:
            print_error(f"Status: {response.status_code}")
            print_error(f"Response: {response.text[:200]}")
            return False, None
            
    except requests.exceptions.ConnectionError:
        print_error("Connection Error!")
        print_error("Make sure FastAPI server is running:")
        print_error("  python main.py")
        return False, None
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False, None

def main():
    print(f"{Fore.MAGENTA}")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "STOCK PREDICTION API TESTING" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    print(Style.RESET_ALL)
    
    results = []
    
    # Test 1: Root Health Check
    print_header("TEST 1: Root Health Check")
    success, _ = test_endpoint("GET", "/", "Root endpoint")
    results.append(("Root Health Check", success))
    
    # Test 2: Detailed Health
    print_header("TEST 2: Detailed Health Check")
    success, _ = test_endpoint("GET", "/health", "Detailed health with DB status")
    results.append(("Detailed Health", success))
    
    # Test 3: Market Data
    print_header("TEST 3: Market Data")
    success, market_data = test_endpoint("GET", "/api/market", "Get all market data")
    results.append(("Market Data", success))
    
    if success and market_data:
        print_info(f"Found {market_data.get('count', 0)} stocks in market")
    
    # Test 4: All Stocks
    print_header("TEST 4: All Stocks Summary")
    success, stocks_data = test_endpoint("GET", "/api/stocks", "Get all stocks with predictions")
    results.append(("All Stocks", success))
    
    ticker = None
    if success and stocks_data:
        count = stocks_data.get('count', 0)
        print_info(f"Found {count} stocks with predictions")
        if count > 0:
            ticker = stocks_data['data'][0]['ticker']
            print_info(f"Using ticker for detailed tests: {ticker}")
    
    # Test 5: Stock Detail
    if ticker:
        print_header("TEST 5: Stock Detail")
        success, stock_detail = test_endpoint("GET", f"/api/stocks/{ticker}", f"Get detail for {ticker}")
        results.append(("Stock Detail", success))
        
        if success and stock_detail:
            data = stock_detail['data']
            print_info(f"Current: Rp {data['price']['current']:,.2f}")
            print_info(f"Pred 1w: Rp {data['predictions']['1w']['price']:,.2f} ({data['predictions']['1w']['change_pct']:+.2f}%)")
            print_info(f"Pred 1m: Rp {data['predictions']['1m']['price']:,.2f} ({data['predictions']['1m']['change_pct']:+.2f}%)")
            print_info(f"RSI: {data['technical_indicators']['rsi']:.2f}")
    
    # Test 6: News
    if ticker:
        print_header("TEST 6: News Feed")
        success, news_data = test_endpoint("GET", f"/api/news/{ticker}?limit=10", f"Get news for {ticker}")
        results.append(("News Feed", success))
        
        if success and news_data:
            count = news_data.get('count', 0)
            print_info(f"Found {count} news articles")
    
    # Test 7: News Distribution
    if ticker:
        print_header("TEST 7: News Sentiment Distribution")
        success, dist_data = test_endpoint("GET", f"/api/news/{ticker}/distribution", f"Get sentiment distribution")
        results.append(("News Distribution", success))
    
    # Test 8: Accuracy
    if ticker:
        print_header("TEST 8: Prediction Accuracy")
        success, accuracy_data = test_endpoint("GET", f"/api/accuracy/{ticker}", f"Get accuracy for {ticker}")
        results.append(("Accuracy", success))
        
        if success and accuracy_data:
            data = accuracy_data.get('data', {})
            if 'error' not in data:
                print_info(f"MAPE: {data.get('avg_mape', 0):.2f}%")
                print_info(f"Direction Acc: {data.get('direction_accuracy', 0)*100:.1f}%")
    
    # Test 9: Model Performance
    if ticker:
        print_header("TEST 9: Model Performance")
        success, performance_data = test_endpoint("GET", f"/api/performance/{ticker}", f"Get performance for {ticker}")
        results.append(("Model Performance", success))
        
        if success and performance_data:
            data = performance_data['data']
            print_info(f"Overall MAPE: {data['overall']['avg_mape']:.2f}%")
    
    # Summary
    print_header("TEST SUMMARY")
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        color = Fore.GREEN if success else Fore.RED
        print(f"{color}{test_name:30} : {status}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    
    if passed_tests == total_tests:
        print(f"{Fore.GREEN}🎉 ALL TESTS PASSED! API is working correctly.{Style.RESET_ALL}\n")
        return 0
    else:
        print(f"{Fore.YELLOW}⚠️  Some tests failed. Check the errors above.{Style.RESET_ALL}\n")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Testing interrupted by user{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Unexpected error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)