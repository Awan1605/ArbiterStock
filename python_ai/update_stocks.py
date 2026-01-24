import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python_ai.services.stock_service import update_stock_detail


if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOG']
    results = update_stock_detail(tickers)
    print(results)
