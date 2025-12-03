<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arbiterstock - Real-time Investment Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.1/css/buttons.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/responsive/2.5.0/css/responsive.dataTables.min.css">
    
    <style>
        :root {
            --primary-color: #3b82f6;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        }

        .card-hover {
            transition: all 0.3s ease;
            border: 1px solid transparent;
        }

        .card-hover:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            border-color: rgba(59, 130, 246, 0.1);
        }

        .gradient-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .gradient-success {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        }

        .gradient-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }

        .stock-row {
            cursor: pointer;
            transition: background-color 0.2s ease;
        }

        .stock-row:hover {
            background-color: #f8fafc;
        }

        /* Custom DataTables styling */
        .dataTables_wrapper {
            padding: 0 !important;
        }

        .dataTables_wrapper .dataTables_filter input {
            border: 1px solid #d1d5db;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            margin-left: 0;
        }

        .dataTables_wrapper .dataTables_filter input:focus {
            outline: none;
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }

        .dataTables_wrapper .dataTables_paginate .paginate_button {
            border: 1px solid #d1d5db !important;
            border-radius: 0.375rem !important;
            margin: 0 2px !important;
        }

        .dataTables_wrapper .dataTables_paginate .paginate_button.current {
            background: #3b82f6 !important;
            color: white !important;
            border-color: #3b82f6 !important;
        }

        /* Symbol badges */
        .symbol-badge {
            font-family: 'Inter', monospace;
            font-weight: 700;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
        }

        .symbol-badge:hover {
            transform: scale(1.05);
        }

        /* Subtle loading indicator */
        .loading-indicator {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #3b82f6);
            background-size: 200% 100%;
            animation: loading-line 2s infinite;
            z-index: 9999;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .loading-indicator.active {
            opacity: 1;
        }

        @keyframes loading-line {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* Skeleton loading */
        .skeleton {
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
        }

        @keyframes loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
    </style>
</head>

<body class="bg-gray-50 text-gray-900">
    <!-- Loading Indicator -->
    <div id="loadingIndicator" class="loading-indicator"></div>

    <!-- Navbar -->
    <x-navbar />

    <main class="container mx-auto px-4 py-8">
        <!-- Market Statistics -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <!-- S&P 500 -->
            <div class="bg-white rounded-xl shadow-lg p-6 border border-gray-200 card-hover">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-gray-500 text-sm">S&P 500 Index</p>
                        <p class="text-2xl font-bold mt-1" id="sp500-price">--</p>
                    </div>
                    <div id="sp500-change" class="px-3 py-1 rounded-full text-xs font-semibold flex items-center skeleton" style="width: 80px; height: 28px;"></div>
                </div>
                <div class="mt-4">
                    <div class="flex items-center justify-between text-sm">
                        <span class="text-gray-500">Today's Change</span>
                        <span id="sp500-change-value" class="font-medium">--</span>
                    </div>
                    <div class="flex items-center justify-between text-sm mt-2">
                        <span class="text-gray-500">52 Week Range</span>
                        <span id="sp500-range" class="font-medium">--</span>
                    </div>
                </div>
            </div>
            
            <!-- Dow Jones -->
            <div class="bg-white rounded-xl shadow-lg p-6 border border-gray-200 card-hover">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-gray-500 text-sm">Dow Jones Industrial</p>
                        <p class="text-2xl font-bold mt-1" id="dowjones-price">--</p>
                    </div>
                    <div id="dowjones-change" class="px-3 py-1 rounded-full text-xs font-semibold flex items-center skeleton" style="width: 80px; height: 28px;"></div>
                </div>
                <div class="mt-4">
                    <div class="flex items-center justify-between text-sm">
                        <span class="text-gray-500">Today's Change</span>
                        <span id="dowjones-change-value" class="font-medium">--</span>
                    </div>
                    <div class="flex items-center justify-between text-sm mt-2">
                        <span class="text-gray-500">52 Week Range</span>
                        <span id="dowjones-range" class="font-medium">--</span>
                    </div>
                </div>
            </div>
            
            <!-- NASDAQ -->
            <div class="bg-white rounded-xl shadow-lg p-6 border border-gray-200 card-hover">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-gray-500 text-sm">NASDAQ Composite</p>
                        <p class="text-2xl font-bold mt-1" id="nasdaq-price">--</p>
                    </div>
                    <div id="nasdaq-change" class="px-3 py-1 rounded-full text-xs font-semibold flex items-center skeleton" style="width: 80px; height: 28px;"></div>
                </div>
                <div class="mt-4">
                    <div class="flex items-center justify-between text-sm">
                        <span class="text-gray-500">Today's Change</span>
                        <span id="nasdaq-change-value" class="font-medium">--</span>
                    </div>
                    <div class="flex items-center justify-between text-sm mt-2">
                        <span class="text-gray-500">52 Week Range</span>
                        <span id="nasdaq-range" class="font-medium">--</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Top Markets Section -->
        <section class="mb-12">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold">Top Markets</h2>
                <div class="flex items-center space-x-2">
                    <span class="text-sm text-gray-500" id="lastUpdate">Last update: --</span>
                    <button id="refreshData" class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center ml-4">
                        <i class="fas fa-sync-alt mr-2"></i> Refresh
                    </button>
                    <div id="refreshLoading" class="hidden ml-2">
                        <div class="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                    </div>
                </div>
            </div>

            <div class="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
                <!-- Table -->
                <div class="overflow-x-auto">
                    <table id="topMarketsTable" class="w-full display nowrap" style="width:100%">
                        <thead>
                            <tr class="border-b border-gray-200 text-gray-500 font-semibold text-sm bg-gray-50">
                                <th class="px-6 py-4 text-left">Symbol</th>
                                <th class="px-6 py-4 text-right">Last Price</th>
                                <th class="px-6 py-4 text-right">Change (%)</th>
                                <th class="px-6 py-4 text-right">Change (Value)</th>
                                <th class="px-6 py-4 text-right">Volume</th>
                                <th class="px-6 py-4 text-right">Day High</th>
                                <th class="px-6 py-4 text-right">Day Low</th>
                                <th class="px-6 py-4 text-right">Open</th>
                                <th class="px-6 py-4 text-right">Prev Close</th>
                            </tr>
                        </thead>
                        <tbody id="stocksTableBody">
                            <!-- Initial loading skeleton -->
                            <tr>
                                <td colspan="9" class="px-6 py-8 text-center text-gray-500">
                                    <div class="flex justify-center">
                                        <div class="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                                    </div>
                                    <p class="mt-2">Loading market data...</p>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <!-- Table Info -->
                <div class="px-6 py-4 border-t border-gray-200 bg-gray-50 flex flex-col sm:flex-row justify-between items-center text-sm text-gray-500">
                    <div class="flex items-center">
                        <i class="fas fa-info-circle mr-2 text-blue-500"></i>
                        <span>Data provided by Yahoo Finance API</span>
                    </div>
                    <div class="mt-2 sm:mt-0 flex items-center space-x-2">
                        <span id="stockCount">Loading stocks...</span>
                        <div id="tableLoading" class="hidden">
                            <div class="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- Today Top Movers -->
        <section class="mb-12">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold">Today Top Movers</h2>
                <a href="#" class="text-blue-600 text-sm font-medium flex items-center hover:text-blue-800 transition-colors">
                    View All <i class="fas fa-chevron-right ml-1 text-xs"></i>
                </a>
            </div>
            <div id="topMoversContainer" class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <!-- Data will be populated by JavaScript -->
                <div class="bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover skeleton" style="height: 120px;"></div>
                <div class="bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover skeleton" style="height: 120px;"></div>
                <div class="bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover skeleton" style="height: 120px;"></div>
                <div class="bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover skeleton" style="height: 120px;"></div>
                <div class="bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover skeleton" style="height: 120px;"></div>
                <div class="bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover skeleton" style="height: 120px;"></div>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer class="bg-white border-t border-gray-200 py-8 mt-12">
        <div class="container mx-auto px-4">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
                <div>
                    <div class="flex items-center space-x-3 mb-4">
                        <div class="w-10 h-10 gradient-primary rounded-lg flex items-center justify-center">
                            <i class="fas fa-chart-line text-white text-xl"></i>
                        </div>
                        <h4 class="font-bold text-lg">Arbiterstock</h4>
                    </div>
                    <p class="text-gray-600 text-sm">Advanced investment platform with real-time market data.</p>
                </div>
                <div>
                    <h4 class="font-semibold mb-4">Markets</h4>
                    <ul class="space-y-2 text-sm text-gray-600">
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Stocks</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Crypto</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Forex</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Commodities</a></li>
                    </ul>
                </div>
                <div>
                    <h4 class="font-semibold mb-4">Resources</h4>
                    <ul class="space-y-2 text-sm text-gray-600">
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Market Analysis</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Economic Calendar</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Learning Center</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">API Documentation</a></li>
                    </ul>
                </div>
                <div>
                    <h4 class="font-semibold mb-4">Support</h4>
                    <ul class="space-y-2 text-sm text-gray-600">
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Help Center</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Contact Us</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">System Status</a></li>
                    </ul>
                </div>
            </div>
            <div class="pt-8 border-t border-gray-200 flex flex-col md:flex-row justify-between items-center">
                <div class="flex space-x-6 text-sm text-gray-500 mb-4 md:mb-0">
                    <a href="#" class="hover:text-gray-700 transition-colors">Terms of Service</a>
                    <a href="#" class="hover:text-gray-700 transition-colors">Privacy Policy</a>
                    <a href="#" class="hover:text-gray-700 transition-colors">Cookie Policy</a>
                </div>
                <div class="text-gray-500 text-sm flex items-center">
                    <i class="far fa-copyright mr-1"></i>
                    <span>2024 Arbiterstock. All rights reserved.</span>
                </div>
            </div>
        </div>
    </footer>

    <!-- jQuery & DataTables JS -->
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.1/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/responsive/2.5.0/js/dataTables.responsive.min.js"></script>

    <script>
        // Configuration
        const CONFIG = {
            symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ', 'WMT', 'PG', 'DIS', 'NFLX', 'ADBE'],
            indices: {
                '^GSPC': { name: 'S&P 500', element: 'sp500' },
                '^DJI': { name: 'Dow Jones', element: 'dowjones' },
                '^IXIC': { name: 'NASDAQ', element: 'nasdaq' }
            },
            updateInterval: 60000 // 1 minute
        };

        let stockDataTable = null;
        let isRefreshing = false;
        let lastUpdateTime = null;

        // Utility functions
        function formatNumber(num) {
            if (!num) return '--';
            if (num >= 1e12) return '$' + (num / 1e12).toFixed(2) + 'T';
            if (num >= 1e9) return '$' + (num / 1e9).toFixed(2) + 'B';
            if (num >= 1e6) return '$' + (num / 1e6).toFixed(2) + 'M';
            if (num >= 1e3) return '$' + (num / 1e3).toFixed(2) + 'K';
            return '$' + num.toFixed(2);
        }

        function formatPrice(price) {
            if (!price) return '--';
            return '$' + price.toFixed(2);
        }

        function formatPercent(percent) {
            if (percent === null || percent === undefined) return '--';
            const sign = percent >= 0 ? '+' : '';
            return sign + percent.toFixed(2) + '%';
        }

        function formatChangeValue(change) {
            if (!change) return '--';
            const sign = change >= 0 ? '+' : '';
            return sign + '$' + Math.abs(change).toFixed(2);
        }

        function getChangeColor(change) {
            return change >= 0 ? 'text-green-600' : 'text-red-600';
        }

        function getChangeBgColor(change) {
            return change >= 0 ? 'bg-green-50' : 'bg-red-50';
        }

        function getChangeIcon(change) {
            return change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
        }

        function getSymbolColor(symbol) {
            // Assign consistent colors based on symbol
            const colors = {
                'AAPL': 'bg-gradient-to-r from-gray-800 to-gray-900 text-white',
                'MSFT': 'bg-gradient-to-r from-blue-500 to-blue-700 text-white',
                'GOOGL': 'bg-gradient-to-r from-red-500 to-red-600 text-white',
                'AMZN': 'bg-gradient-to-r from-yellow-400 to-yellow-600 text-gray-900',
                'TSLA': 'bg-gradient-to-r from-red-500 to-red-700 text-white',
                'META': 'bg-gradient-to-r from-blue-600 to-blue-800 text-white',
                'NVDA': 'bg-gradient-to-r from-green-500 to-green-700 text-white',
                'JPM': 'bg-gradient-to-r from-blue-600 to-blue-800 text-white',
                'V': 'bg-gradient-to-r from-yellow-500 to-yellow-700 text-white',
                'JNJ': 'bg-gradient-to-r from-red-400 to-red-600 text-white',
                'WMT': 'bg-gradient-to-r from-blue-600 to-blue-800 text-white',
                'PG': 'bg-gradient-to-r from-blue-400 to-blue-600 text-white',
                'DIS': 'bg-gradient-to-r from-blue-400 to-blue-600 text-white',
                'NFLX': 'bg-gradient-to-r from-red-600 to-red-800 text-white',
                'ADBE': 'bg-gradient-to-r from-red-400 to-red-600 text-white'
            };
            return colors[symbol] || 'bg-gradient-to-r from-indigo-500 to-indigo-700 text-white';
        }

        function getCompanyName(symbol) {
            const companyNames = {
                'AAPL': 'Apple Inc.',
                'MSFT': 'Microsoft Corporation',
                'GOOGL': 'Alphabet Inc.',
                'AMZN': 'Amazon.com Inc.',
                'TSLA': 'Tesla Inc.',
                'META': 'Meta Platforms Inc.',
                'NVDA': 'NVIDIA Corporation',
                'JPM': 'JPMorgan Chase & Co.',
                'V': 'Visa Inc.',
                'JNJ': 'Johnson & Johnson',
                'WMT': 'Walmart Inc.',
                'PG': 'Procter & Gamble Co.',
                'DIS': 'Walt Disney Company',
                'NFLX': 'Netflix Inc.',
                'ADBE': 'Adobe Inc.'
            };
            return companyNames[symbol] || `${symbol}`;
        }

        // Loading management
        function showLoading(indicator = true, table = false) {
            if (indicator) {
                document.getElementById('loadingIndicator').classList.add('active');
            }
            if (table) {
                document.getElementById('tableLoading').classList.remove('hidden');
            }
            isRefreshing = true;
            document.getElementById('refreshLoading').classList.remove('hidden');
            document.getElementById('refreshData').classList.add('opacity-50', 'cursor-not-allowed');
        }

        function hideLoading(indicator = true, table = false) {
            if (indicator) {
                document.getElementById('loadingIndicator').classList.remove('active');
            }
            if (table) {
                document.getElementById('tableLoading').classList.add('hidden');
            }
            isRefreshing = false;
            document.getElementById('refreshLoading').classList.add('hidden');
            document.getElementById('refreshData').classList.remove('opacity-50', 'cursor-not-allowed');
        }

        // API functions
        async function fetchStockData(symbols) {
            const promises = symbols.map(async (symbol) => {
                try {
                    const response = await fetch(`https://api.allorigins.win/get?url=${encodeURIComponent(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=1d&interval=1m`)}`);
                    const data = await response.json();
                    const chartData = JSON.parse(data.contents);
                    
                    if (chartData.chart?.result?.[0]) {
                        const result = chartData.chart.result[0];
                        const meta = result.meta;
                        const indicators = result.indicators.quote[0];
                        
                        // Get day high/low from today's data
                        const todayHigh = Math.max(...indicators.high.filter(h => h));
                        const todayLow = Math.min(...indicators.low.filter(l => l && l > 0));
                        const todayOpen = indicators.open.find(o => o);
                        const volume = indicators.volume.reduce((sum, vol) => sum + (vol || 0), 0);
                        
                        return {
                            symbol,
                            price: meta.regularMarketPrice,
                            change: meta.regularMarketPrice - meta.previousClose,
                            changePercent: ((meta.regularMarketPrice - meta.previousClose) / meta.previousClose) * 100,
                            volume: volume,
                            dayHigh: todayHigh || meta.regularMarketDayHigh || meta.regularMarketPrice,
                            dayLow: todayLow || meta.regularMarketDayLow || meta.regularMarketPrice,
                            open: todayOpen || meta.regularMarketOpen || meta.previousClose,
                            previousClose: meta.previousClose
                        };
                    }
                } catch (error) {
                    console.error(`Error fetching data for ${symbol}:`, error);
                }
                return null;
            });

            const results = await Promise.all(promises);
            return results.filter(item => item !== null);
        }

        async function fetchIndexData() {
            const results = {};
            
            for (const [symbol, config] of Object.entries(CONFIG.indices)) {
                try {
                    const response = await fetch(`https://api.allorigins.win/get?url=${encodeURIComponent(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=1d&interval=1m`)}`);
                    const data = await response.json();
                    const chartData = JSON.parse(data.contents);
                    
                    if (chartData.chart?.result?.[0]) {
                        const result = chartData.chart.result[0];
                        const meta = result.meta;
                        
                        results[symbol] = {
                            price: meta.regularMarketPrice,
                            change: meta.regularMarketPrice - meta.previousClose,
                            changePercent: ((meta.regularMarketPrice - meta.previousClose) / meta.previousClose) * 100,
                            dayHigh: meta.regularMarketDayHigh,
                            dayLow: meta.regularMarketDayLow,
                            fiftyTwoWeekHigh: meta.fiftyTwoWeekHigh,
                            fiftyTwoWeekLow: meta.fiftyTwoWeekLow
                        };
                    }
                } catch (error) {
                    console.error(`Error fetching index data for ${symbol}:`, error);
                }
            }
            
            return results;
        }

        // UI update functions
        function updateIndexCards(indexData) {
            Object.entries(indexData).forEach(([symbol, data]) => {
                const config = CONFIG.indices[symbol];
                if (config && data.price) {
                    // Update price
                    document.getElementById(`${config.element}-price`).textContent = formatPrice(data.price);
                    
                    // Remove skeleton loading
                    const changeElement = document.getElementById(`${config.element}-change`);
                    if (changeElement.classList.contains('skeleton')) {
                        changeElement.classList.remove('skeleton');
                    }
                    
                    // Update change percentage
                    changeElement.className = `${getChangeColor(data.changePercent)} ${getChangeBgColor(data.changePercent)} px-3 py-1 rounded-full text-xs font-semibold flex items-center`;
                    changeElement.innerHTML = `<i class="fas ${getChangeIcon(data.changePercent)} mr-1"></i> ${formatPercent(data.changePercent)}`;
                    
                    // Update change value
                    document.getElementById(`${config.element}-change-value`).textContent = formatChangeValue(data.change);
                    
                    // Update 52 week range
                    const rangeElement = document.getElementById(`${config.element}-range`);
                    if (data.fiftyTwoWeekHigh && data.fiftyTwoWeekLow) {
                        rangeElement.textContent = `${formatPrice(data.fiftyTwoWeekLow)} - ${formatPrice(data.fiftyTwoWeekHigh)}`;
                    } else {
                        rangeElement.textContent = '--';
                    }
                }
            });
        }

        function updateStocksTable(stocks) {
            const tbody = document.getElementById('stocksTableBody');
            tbody.innerHTML = '';

            stocks.forEach(stock => {
                const row = document.createElement('tr');
                row.className = 'stock-row border-b border-gray-100 hover:bg-gray-50 transition-all duration-300';
                row.setAttribute('data-symbol', stock.symbol);
                row.onclick = () => window.location.href = `/stocks/${stock.symbol}`;
                
                // Animate row appearance
                row.style.opacity = '0';
                row.style.transform = 'translateY(10px)';
                
                row.innerHTML = `
                    <td class="px-6 py-4">
                        <div class="flex items-center space-x-3">
                            <div class="w-12 h-12 rounded-xl flex items-center justify-center shadow-sm ${getSymbolColor(stock.symbol)} symbol-badge">
                                <span class="font-bold text-sm">${stock.symbol}</span>
                            </div>
                            <div>
                                <p class="font-bold text-gray-800">${stock.symbol}</p>
                                <p class="text-gray-500 text-sm">${getCompanyName(stock.symbol)}</p>
                            </div>
                        </div>
                    </td>
                    <td class="px-6 py-4 text-right font-bold text-gray-900">
                        ${formatPrice(stock.price)}
                    </td>
                    <td class="px-6 py-4 text-right">
                        <span class="inline-flex items-center ${getChangeColor(stock.changePercent)} ${getChangeBgColor(stock.changePercent)} px-3 py-1 rounded-full font-semibold text-sm transition-all duration-300">
                            <i class="fas ${getChangeIcon(stock.changePercent)} mr-1 text-xs"></i> ${formatPercent(stock.changePercent)}
                        </span>
                    </td>
                    <td class="px-6 py-4 text-right font-semibold ${getChangeColor(stock.change)}">
                        ${formatChangeValue(stock.change)}
                    </td>
                    <td class="px-6 py-4 text-right text-gray-700 font-medium">
                        ${stock.volume ? formatNumber(stock.volume) : '--'}
                    </td>
                    <td class="px-6 py-4 text-right text-gray-700">
                        ${formatPrice(stock.dayHigh)}
                    </td>
                    <td class="px-6 py-4 text-right text-gray-700">
                        ${formatPrice(stock.dayLow)}
                    </td>
                    <td class="px-6 py-4 text-right text-gray-700">
                        ${formatPrice(stock.open)}
                    </td>
                    <td class="px-6 py-4 text-right text-gray-700">
                        ${formatPrice(stock.previousClose)}
                    </td>
                `;
                tbody.appendChild(row);
                
                // Animate row appearance
                setTimeout(() => {
                    row.style.opacity = '1';
                    row.style.transform = 'translateY(0)';
                }, 10);
            });

            // Initialize or update DataTable
            if (!stockDataTable) {
                stockDataTable = $('#topMarketsTable').DataTable({
                    responsive: true,
                    paging: true,
                    pageLength: 10,
                    lengthChange: false,
                    searching: true,
                    ordering: true,
                    info: true,
                    autoWidth: false,
                    order: [[1, 'desc']], // Sort by Last Price descending by default
                    language: {
                        search: "",
                        searchPlaceholder: "Search stocks...",
                        emptyTable: "No stock data available",
                        zeroRecords: "No matching stocks found",
                        info: "Showing _START_ to _END_ of _TOTAL_ stocks",
                        infoEmpty: "Showing 0 to 0 of 0 stocks",
                        infoFiltered: "(filtered from _MAX_ total stocks)"
                    },
                    dom: '<"top"f>rt<"bottom"lp><"clear">',
                    columnDefs: [
                        { responsivePriority: 1, targets: 0 }, // Symbol
                        { responsivePriority: 2, targets: 1 }, // Price
                        { responsivePriority: 3, targets: 2 }, // Change %
                        { responsivePriority: 4, targets: 3 }, // Change Value
                        { responsivePriority: 5, targets: 4 }, // Volume
                        { responsivePriority: 6, targets: 5 }, // Day High
                        { responsivePriority: 7, targets: 6 }, // Day Low
                        { responsivePriority: 8, targets: 7 }, // Open
                        { responsivePriority: 9, targets: 8 }  // Prev Close
                    ]
                });
            } else {
                stockDataTable.clear();
                stockDataTable.rows.add($('#stocksTableBody tr'));
                stockDataTable.draw();
            }

            document.getElementById('stockCount').textContent = `Showing ${stocks.length} stocks`;
        }

        function updateTopMovers(stocks) {
            const container = document.getElementById('topMoversContainer');
            
            // Clear existing skeletons
            container.innerHTML = '';

            // Sort by absolute percentage change and take top 6
            const topMovers = [...stocks]
                .sort((a, b) => Math.abs(b.changePercent) - Math.abs(a.changePercent))
                .slice(0, 6);

            topMovers.forEach((stock, index) => {
                const card = document.createElement('div');
                card.className = 'bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover';
                card.setAttribute('data-symbol', stock.symbol);
                card.style.cursor = 'pointer';
                card.onclick = () => window.location.href = `/stocks/${stock.symbol}`;
                
                // Add animation delay for staggered appearance
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.style.animationDelay = `${index * 100}ms`;
                card.style.animation = 'fadeInUp 0.5s ease forwards';
                
                card.innerHTML = `
                    <div class="flex justify-between items-start mb-3">
                        <div class="flex items-center">
                            <div class="w-10 h-10 rounded-lg flex items-center justify-center shadow-sm mr-2 ${getSymbolColor(stock.symbol)} symbol-badge">
                                <span class="font-bold text-xs">${stock.symbol}</span>
                            </div>
                            <div>
                                <p class="font-bold text-sm">${stock.symbol}</p>
                                <p class="text-gray-500 text-xs" style="max-width: 80px;">${getCompanyName(stock.symbol)}</p>
                            </div>
                        </div>
                    </div>
                    <div class="mt-4">
                        <p class="font-bold text-lg text-gray-900">${formatPrice(stock.price)}</p>
                        <div class="mt-2 flex items-center justify-between">
                            <span class="${getChangeColor(stock.changePercent)} font-semibold text-sm">
                                ${formatPercent(stock.changePercent)}
                            </span>
                            <span class="${getChangeColor(stock.change)} text-xs font-medium flex items-center">
                                <i class="fas ${getChangeIcon(stock.change)} mr-1"></i> ${formatChangeValue(stock.change)}
                            </span>
                        </div>
                    </div>
                `;
                container.appendChild(card);
                
                // Trigger animation
                setTimeout(() => {
                    card.style.opacity = '1';
                    card.style.transform = 'translateY(0)';
                }, index * 100);
            });
        }

        // Main update function
        async function updateAllData(showLoadingIndicator = false) {
            if (isRefreshing) return;
            
            try {
                if (showLoadingIndicator) {
                    showLoading(true, true);
                }
                
                const [stockData, indexData] = await Promise.all([
                    fetchStockData(CONFIG.symbols),
                    fetchIndexData()
                ]);

                // Sort stocks by price (descending)
                stockData.sort((a, b) => (b.price || 0) - (a.price || 0));
                
                updateStocksTable(stockData);
                updateTopMovers(stockData);
                updateIndexCards(indexData);
                
                // Update timestamp
                lastUpdateTime = new Date();
                document.getElementById('lastUpdate').textContent = `Last update: ${lastUpdateTime.toLocaleTimeString()}`;
                
            } catch (error) {
                console.error('Error updating data:', error);
                // Show error in a non-intrusive way
                const errorElement = document.createElement('div');
                errorElement.className = 'fixed bottom-4 right-4 bg-red-50 border border-red-200 text-red-600 px-4 py-2 rounded-lg shadow-lg text-sm z-50';
                errorElement.textContent = 'Failed to update data. Retrying...';
                document.body.appendChild(errorElement);
                
                setTimeout(() => {
                    errorElement.remove();
                }, 3000);
            } finally {
                if (showLoadingIndicator) {
                    hideLoading(true, true);
                }
            }
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            // Initial load without loading indicator
            updateAllData(false);
            
            // Set up auto-refresh (silent refresh without indicator)
            setInterval(() => {
                updateAllData(false);
            }, CONFIG.updateInterval);
            
            // Refresh button with loading indicator
            document.getElementById('refreshData').addEventListener('click', () => {
                updateAllData(true);
            });
            
            // Handle row clicks for stock detail
            document.addEventListener('click', function(e) {
                const row = e.target.closest('.stock-row');
                if (row) {
                    const symbol = row.getAttribute('data-symbol');
                    window.location.href = `/stocks/${symbol}`;
                }
                
                const card = e.target.closest('[data-symbol]');
                if (card && card !== document.getElementById('topMoversContainer')) {
                    const symbol = card.getAttribute('data-symbol');
                    window.location.href = `/stocks/${symbol}`;
                }
            });
            
            // Add CSS for animations
            const style = document.createElement('style');
            style.textContent = `
                @keyframes fadeInUp {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                .refresh-pulse {
                    animation: pulse 2s infinite;
                }
                
                @keyframes pulse {
                    0%, 100% {
                        opacity: 1;
                    }
                    50% {
                        opacity: 0.5;
                    }
                }
            `;
            document.head.appendChild(style);
        });
    </script>
</body>
</html>