<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arbiterstock - Advanced Investment Platform</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.13.6/css/dataTables.tailwindcss.min.css">
    <script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js">
    </script>
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

        .line-clamp-2 {
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .scrollbar-thin::-webkit-scrollbar {
            height: 6px;
        }

        .scrollbar-thin::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }

        .scrollbar-thin::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 10px;
        }

        .scrollbar-thin::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }

        .chart-container {
            position: relative;
            height: 60px;
            width: 100%;
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

        .gradient-warning {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        }

        .gradient-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        }

        .glass-effect {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }

        .pulse-animation {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% {
                transform: scale(1);
            }

            50% {
                transform: scale(1.05);
            }

            100% {
                transform: scale(1);
            }
        }

        /* Custom styling for DataTables to match Tailwind */
        .dataTables_wrapper .dataTables_filter input {
            border: 1px solid #d1d5db;
            border-radius: 0.5rem;
            padding: 0.5rem 0.75rem 0.5rem 2.5rem;
            margin-left: 0;
        }

        .dataTables_wrapper .dataTables_filter input:focus {
            outline: none;
            ring: 2px;
            ring-color: #3b82f6;
            border-color: #3b82f6;
        }

        /* Responsive table styling */
        @media (max-width: 768px) {
            #topMarketsTable thead {
                display: none;
            }

            #topMarketsTable,
            #topMarketsTable tbody,
            #topMarketsTable tr,
            #topMarketsTable td {
                display: block;
                width: 100%;
            }

            #topMarketsTable tr {
                margin-bottom: 1rem;
                border: 1px solid #e5e7eb;
                border-radius: 0.5rem;
                padding: 1rem;
            }

            #topMarketsTable td {
                text-align: right;
                padding-left: 50%;
                position: relative;
                border-bottom: 1px solid #f3f4f6;
            }

            #topMarketsTable td:last-child {
                border-bottom: none;
            }

            #topMarketsTable td::before {
                content: attr(data-label);
                position: absolute;
                left: 1rem;
                width: calc(50% - 1rem);
                padding-right: 1rem;
                font-weight: 600;
                text-align: left;
                color: #6b7280;
            }

            /* Add data labels for mobile view */
            #topMarketsTable td:nth-child(1)::before {
                content: "Symbol";
            }

            #topMarketsTable td:nth-child(2)::before {
                content: "24H Change";
            }

            #topMarketsTable td:nth-child(3)::before {
                content: "1W Change";
            }

            #topMarketsTable td:nth-child(4)::before {
                content: "1M Change";
            }

            #topMarketsTable td:nth-child(5)::before {
                content: "Market Cap";
            }

            #topMarketsTable td:nth-child(6)::before {
                content: "Current Price";
            }
        }
    </style>
</head>

<body class="bg-gray-50 text-gray-900">
    <!-- Navbar -->
    <x-navbar></x-navbar>

    <main class="container mx-auto px-4 py-8">
        <!-- Market Statistics -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <!-- S&P 500 -->
            <div class="bg-white rounded-xl shadow-lg p-6 border border-gray-200 card-hover">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-gray-500 text-sm">S&P 500</p>
                        <p class="text-2xl font-bold mt-1">4,567.25</p>
                    </div>
                    <div
                        class="bg-green-100 text-green-600 px-3 py-1 rounded-full text-xs font-semibold flex items-center">
                        <i class="fas fa-arrow-up mr-1"></i> +1.35%
                    </div>
                </div>
                <div class="mt-4 h-12">
                    <canvas id="sp500-chart"></canvas>
                </div>
            </div>

            <!-- Dow Jones -->
            <div class="bg-white rounded-xl shadow-lg p-6 border border-gray-200 card-hover">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-gray-500 text-sm">Dow Jones</p>
                        <p class="text-2xl font-bold mt-1">35,678.42</p>
                    </div>
                    <div
                        class="bg-green-100 text-green-600 px-3 py-1 rounded-full text-xs font-semibold flex items-center">
                        <i class="fas fa-arrow-up mr-1"></i> +0.82%
                    </div>
                </div>
                <div class="mt-4 h-12">
                    <canvas id="dowjones-chart"></canvas>
                </div>
            </div>

            <!-- NASDAQ -->
            <div class="bg-white rounded-xl shadow-lg p-6 border border-gray-200 card-hover">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-gray-500 text-sm">NASDAQ</p>
                        <p class="text-2xl font-bold mt-1">14,235.67</p>
                    </div>
                    <div
                        class="bg-green-100 text-green-600 px-3 py-1 rounded-full text-xs font-semibold flex items-center">
                        <i class="fas fa-arrow-up mr-1"></i> +2.14%
                    </div>
                </div>
                <div class="mt-4 h-12">
                    <canvas id="nasdaq-chart"></canvas>
                </div>
            </div>

            <!-- Market Volatility (VIX) -->
            <div class="bg-white rounded-xl shadow-lg p-6 border border-gray-200 card-hover">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-gray-500 text-sm">VIX Index</p>
                        <p class="text-2xl font-bold mt-1">15.24</p>
                    </div>
                    <div class="bg-red-100 text-red-600 px-3 py-1 rounded-full text-xs font-semibold flex items-center">
                        <i class="fas fa-arrow-down mr-1"></i> -0.36%
                    </div>
                </div>
                <div class="mt-4 flex items-center">
                    <div class="w-3 h-3 bg-green-500 rounded-full mr-2 pulse-animation"></div>
                    <p class="text-gray-600 text-sm">Low volatility</p>
                </div>
            </div>
        </div>

        <!-- Market Overview -->
        <div class="bg-white rounded-xl shadow-lg p-6 mb-8 border border-gray-200 card-hover">
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-xl font-bold">Market Overview</h2>
                <div class="text-sm text-gray-500">Real-time data</div>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="text-center p-4 rounded-lg bg-green-50 border border-green-100">
                    <p class="text-gray-500 text-sm">Advancing</p>
                    <p class="text-2xl font-bold text-green-600">2,345</p>
                </div>
                <div class="text-center p-4 rounded-lg bg-red-50 border border-red-100">
                    <p class="text-gray-500 text-sm">Declining</p>
                    <p class="text-2xl font-bold text-red-600">1,234</p>
                </div>
                <div class="text-center p-4 rounded-lg bg-gray-50 border border-gray-100">
                    <p class="text-gray-500 text-sm">Unchanged</p>
                    <p class="text-2xl font-bold text-gray-600">567</p>
                </div>
                <div class="text-center p-4 rounded-lg bg-blue-50 border border-blue-100">
                    <p class="text-gray-500 text-sm">Total Volume</p>
                    <p class="text-2xl font-bold text-blue-600">4.2B</p>
                </div>
            </div>
        </div>

        <!-- Top Markets Section -->
        <section class="mb-12">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold">Top Markets</h2>
                <div class="flex space-x-2 bg-gray-100 p-1 rounded-lg">
                    <button
                        class="px-3 py-1 bg-blue-600 text-white rounded-lg text-sm font-medium transition-all">24H</button>
                    <button
                        class="px-3 py-1 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 transition-all">1W</button>
                    <button
                        class="px-3 py-1 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 transition-all">1M</button>
                    <button
                        class="px-3 py-1 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-200 transition-all">1Y</button>
                </div>
            </div>

            <div class="bg-white rounded-xl shadow-lg overflow-hidden border border-gray-200">
                <!-- Table -->
                <div class="overflow-x-auto">
                    <table id="topMarketsTable" class="w-full">
                        <thead>
                            <tr class="border-b border-gray-200 text-gray-500 font-semibold text-sm bg-gray-50">
                                <th class="px-6 py-4 text-left cursor-pointer hover:bg-gray-100 transition-colors">
                                    <div class="flex items-center">
                                        Symbol
                                        <i class="fas fa-sort ml-1 text-gray-400"></i>
                                    </div>
                                </th>
                                <th class="px-6 py-4 text-center cursor-pointer hover:bg-gray-100 transition-colors">
                                    <div class="flex items-center justify-center">
                                        24H
                                        <i class="fas fa-sort ml-1 text-gray-400"></i>
                                    </div>
                                </th>
                                <th class="px-6 py-4 text-center cursor-pointer hover:bg-gray-100 transition-colors">
                                    <div class="flex items-center justify-center">
                                        1W
                                        <i class="fas fa-sort ml-1 text-gray-400"></i>
                                    </div>
                                </th>
                                <th class="px-6 py-4 text-center cursor-pointer hover:bg-gray-100 transition-colors">
                                    <div class="flex items-center justify-center">
                                        1M
                                        <i class="fas fa-sort ml-1 text-gray-400"></i>
                                    </div>
                                </th>
                                <th class="px-6 py-4 text-right cursor-pointer hover:bg-gray-100 transition-colors">
                                    <div class="flex items-center justify-end">
                                        Market Cap
                                        <i class="fas fa-sort ml-1 text-gray-400"></i>
                                    </div>
                                </th>
                                <th class="px-6 py-4 text-right cursor-pointer hover:bg-gray-100 transition-colors">
                                    <div class="flex items-center justify-end">
                                        Current Price
                                        <i class="fas fa-sort ml-1 text-gray-400"></i>
                                    </div>
                                </th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-100">
                            <!-- AAPL Row -->
                            <tr class="hover:bg-blue-50 transition-colors group">
                                <td class="px-6 py-4">
                                    <div class="flex items-center space-x-3">
                                        <div
                                            class="w-10 h-10 bg-gray-200 rounded-lg flex items-center justify-center group-hover:bg-gray-300 transition-colors">
                                            <i class="fab fa-apple text-gray-700 text-xl"></i>
                                        </div>
                                        <div>
                                            <p class="font-semibold">AAPL</p>
                                            <p class="text-gray-500 text-sm">Apple, Inc</p>
                                        </div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-green-500 font-medium bg-green-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-up mr-1 text-xs"></i> +0.51%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-green-500 font-medium bg-green-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-up mr-1 text-xs"></i> +2.34%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-red-500 font-medium bg-red-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-down mr-1 text-xs"></i> -1.23%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-right font-semibold">$2.8T</td>
                                <td class="px-6 py-4 text-right font-bold text-gray-800">$182.94</td>
                            </tr>

                            <!-- META Row -->
                            <tr class="hover:bg-blue-50 transition-colors group">
                                <td class="px-6 py-4">
                                    <div class="flex items-center space-x-3">
                                        <div
                                            class="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center group-hover:bg-blue-200 transition-colors">
                                            <i class="fab fa-facebook-f text-blue-600 text-xl"></i>
                                        </div>
                                        <div>
                                            <p class="font-semibold">META</p>
                                            <p class="text-gray-500 text-sm">Meta Platforms, Inc</p>
                                        </div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-green-500 font-medium bg-green-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-up mr-1 text-xs"></i> +0.26%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-green-500 font-medium bg-green-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-up mr-1 text-xs"></i> +1.89%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-green-500 font-medium bg-green-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-up mr-1 text-xs"></i> +3.45%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-right font-semibold">$895B</td>
                                <td class="px-6 py-4 text-right font-bold text-gray-800">$328.45</td>
                            </tr>

                            <!-- TSLA Row -->
                            <tr class="hover:bg-blue-50 transition-colors group">
                                <td class="px-6 py-4">
                                    <div class="flex items-center space-x-3">
                                        <div
                                            class="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center group-hover:bg-red-200 transition-colors">
                                            <i class="fab fa-tesla text-red-600 text-xl"></i>
                                        </div>
                                        <div>
                                            <p class="font-semibold">TSLA</p>
                                            <p class="text-gray-500 text-sm">Tesla, Inc</p>
                                        </div>
                                    </div>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-red-500 font-medium bg-red-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-down mr-1 text-xs"></i> -0.34%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-red-500 font-medium bg-red-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-down mr-1 text-xs"></i> -2.15%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-center">
                                    <span
                                        class="inline-flex items-center text-red-500 font-medium bg-red-50 px-2 py-1 rounded-full">
                                        <i class="fas fa-arrow-down mr-1 text-xs"></i> -5.42%
                                    </span>
                                </td>
                                <td class="px-6 py-4 text-right font-semibold">$685B</td>
                                <td class="px-6 py-4 text-right font-bold text-gray-800">$248.92</td>
                            </tr>

                            <!-- Additional stocks would follow the same pattern -->
                            <!-- ... -->
                        </tbody>
                    </table>
                </div>

                <!-- Table Info -->
                <div
                    class="px-6 py-4 border-t border-gray-200 bg-gray-50 flex flex-col sm:flex-row justify-between items-center text-sm text-gray-500">
                    <div class="flex items-center">
                        <i class="fas fa-info-circle mr-2 text-blue-500"></i>
                        <span>Showing <span id="visibleStocks">10</span> of <span id="totalStocks">10</span>
                            stocks</span>
                    </div>
                    <div class="flex items-center space-x-2 mt-2 sm:mt-0">
                        <button id="prevPage"
                            class="px-3 py-1 bg-gray-200 rounded-lg hover:bg-gray-300 disabled:opacity-50 transition-colors"
                            disabled>
                            <i class="fas fa-chevron-left"></i>
                        </button>
                        <span class="mx-2">Page <span id="currentPage">1</span></span>
                        <button id="nextPage"
                            class="px-3 py-1 bg-gray-200 rounded-lg hover:bg-gray-300 disabled:opacity-50 transition-colors"
                            disabled>
                            <i class="fas fa-chevron-right"></i>
                        </button>
                    </div>
                </div>
            </div>
        </section>

        <!-- Today Top Movers -->
        <section class="mb-12">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold">Today Top Movers</h2>
                <a href="#"
                    class="text-blue-600 text-sm font-medium flex items-center hover:text-blue-800 transition-colors">
                    View All <i class="fas fa-chevron-right ml-1 text-xs"></i>
                </a>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                <!-- ADBE Card -->
                <div class="bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover">
                    <div class="flex justify-between items-start mb-3">
                        <div class="flex items-center">
                            <div class="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center mr-2">
                                <i class="fab fa-adobe text-red-600 text-sm"></i>
                            </div>
                            <div>
                                <p class="font-semibold text-sm">ADBE</p>
                                <p class="text-gray-500 text-xs">Adobe, Inc</p>
                            </div>
                        </div>
                        <div class="text-green-500 text-xs font-semibold bg-green-100 px-2 py-1 rounded-full">
                            +0.61%
                        </div>
                    </div>
                    <div class="flex justify-between items-end">
                        <div>
                            <p class="font-bold text-lg">$371.07</p>
                            <p class="text-green-500 text-xs flex items-center">
                                <i class="fas fa-arrow-up mr-1"></i> +$2.24
                            </p>
                        </div>
                        <div class="w-16 h-10">
                            <canvas id="mini-chart-adbe"></canvas>
                        </div>
                    </div>
                </div>

                <!-- ABNB Card -->
                <div class="bg-white rounded-xl shadow-lg p-4 border border-gray-200 card-hover">
                    <div class="flex justify-between items-start mb-3">
                        <div class="flex items-center">
                            <div class="w-8 h-8 bg-pink-100 rounded-lg flex items-center justify-center mr-2">
                                <i class="fab fa-airbnb text-pink-500 text-sm"></i>
                            </div>
                            <div>
                                <p class="font-semibold text-sm">ABNB</p>
                                <p class="text-gray-500 text-xs">Airbnb, Inc</p>
                            </div>
                        </div>
                        <div class="text-red-500 text-xs font-semibold bg-red-100 px-2 py-1 rounded-full">
                            -0.67%
                        </div>
                    </div>
                    <div class="flex justify-between items-end">
                        <div>
                            <p class="font-bold text-lg">$132.01</p>
                            <p class="text-red-500 text-xs flex items-center">
                                <i class="fas fa-arrow-down mr-1"></i> -$0.89
                            </p>
                        </div>
                        <div class="w-16 h-10">
                            <canvas id="mini-chart-abnb"></canvas>
                        </div>
                    </div>
                </div>

                <!-- More cards would follow the same pattern -->
                <!-- ... -->
            </div>
        </section>

        <!-- Latest News -->
        <section class="mb-12">
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold">Latest News</h2>
                <a href="#"
                    class="text-blue-600 text-sm font-medium flex items-center hover:text-blue-800 transition-colors">
                    View All <i class="fas fa-chevron-right ml-1 text-xs"></i>
                </a>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <!-- News Item 1 -->
                <div class="bg-white rounded-xl shadow-lg p-5 border border-gray-200 card-hover">
                    <div class="flex items-start mb-4">
                        <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mr-4">
                            <i class="fab fa-facebook text-blue-600"></i>
                        </div>
                        <div>
                            <p class="font-semibold text-sm mb-2 line-clamp-2">
                                Facebook's "Failed" Libra Cryptocurrency Is No Closer to Release
                            </p>
                            <div class="flex items-center space-x-2">
                                <span class="text-blue-700 font-bold text-xs">META</span>
                                <span class="text-green-500 text-xs flex items-center">
                                    <i class="fas fa-arrow-up text-xs mr-1"></i>0.26%
                                </span>
                            </div>
                        </div>
                    </div>
                    <div class="flex items-center text-gray-500 text-xs">
                        <i class="far fa-clock mr-1"></i>
                        <span>Bloomberg · 35m ago</span>
                    </div>
                </div>

                <!-- More news items would follow the same pattern -->
                <!-- ... -->
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
                    <p class="text-gray-600 text-sm">Advanced investment platform for modern traders and investors.</p>
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
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Learning Center</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Market Analysis</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Economic Calendar</a></li>
                        <li><a href="#" class="hover:text-blue-600 transition-colors">Blog</a></li>
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
                    <span>2023 Arbiterstock. All rights reserved.</span>
                </div>
            </div>
        </div>
    </footer>

    <script>
        // Function to generate random stock data with different trends
        function generateStockData(trend, points = 20, volatility = 5) {
            const data = [];
            let value = 50;

            for (let i = 0; i < points; i++) {
                data.push(value);

                // Add some randomness but maintain the trend
                const randomChange = (Math.random() - 0.5) * volatility;

                if (trend === 'up') {
                    value += Math.random() * 2 + randomChange;
                } else if (trend === 'down') {
                    value += (Math.random() - 1) * 2 + randomChange;
                } else {
                    value += randomChange;
                }

                // Ensure values stay within a reasonable range
                value = Math.max(10, Math.min(90, value));
            }

            return data;
        }

        // Function to create a chart
        function createChart(ctx, data, color, fill = false) {
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array(data.length).fill(''),
                    datasets: [{
                        data: data,
                        borderColor: color,
                        borderWidth: 2,
                        fill: fill,
                        backgroundColor: fill ? `${color}20` : 'transparent',
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            enabled: false
                        }
                    },
                    scales: {
                        x: {
                            display: false
                        },
                        y: {
                            display: false
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    elements: {
                        point: {
                            radius: 0
                        }
                    }
                }
            });
        }

        // Initialize all charts when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            // Market index charts
            const sp500Ctx = document.getElementById('sp500-chart').getContext('2d');
            const sp500Data = generateStockData('up', 15, 2);
            createChart(sp500Ctx, sp500Data, '#10B981', true);

            const dowjonesCtx = document.getElementById('dowjones-chart').getContext('2d');
            const dowjonesData = generateStockData('up', 15, 1.5);
            createChart(dowjonesCtx, dowjonesData, '#10B981', true);

            const nasdaqCtx = document.getElementById('nasdaq-chart').getContext('2d');
            const nasdaqData = generateStockData('up', 15, 3);
            createChart(nasdaqCtx, nasdaqData, '#10B981', true);

            // Initialize DataTable
            $(document).ready(function() {
                var table = $('#topMarketsTable').DataTable({
                    "paging": true,
                    "searching": true,
                    "info": false,
                    "lengthChange": false,
                    "pageLength": 10,
                    "order": [
                        [0, 'asc']
                    ],
                    "language": {
                        "search": "",
                        "searchPlaceholder": "Search stocks...",
                        "emptyTable": "No stocks found",
                        "zeroRecords": "No matching stocks found"
                    },
                    "dom": '<"top"f>rt<"bottom"lp><"clear">',
                    "initComplete": function() {
                        // Move search box to our custom location
                        $('.dataTables_filter').appendTo('#stockSearch').parent();
                        $('.dataTables_filter input').addClass(
                            'pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full sm:w-64'
                        );
                        $('.dataTables_filter').html(
                            '<div class="relative"><div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none"><i class="fas fa-search text-gray-400"></i></div>' +
                            $('.dataTables_filter').html() + '</div>');

                        // Update stock count
                        updateStockCount();
                    },
                    "drawCallback": function() {
                        updateStockCount();
                    }
                });

                // Custom search functionality
                $('#stockSearch').on('keyup', function() {
                    table.search(this.value).draw();
                });

                // Custom pagination
                $('#prevPage').on('click', function() {
                    table.page('previous').draw('page');
                    updatePagination();
                });

                $('#nextPage').on('click', function() {
                    table.page('next').draw('page');
                    updatePagination();
                });

                // Update stock count
                function updateStockCount() {
                    var total = table.data().count();
                    var filtered = table.rows({
                        search: 'applied'
                    }).count();
                    var visible = table.page.info().recordsDisplay;

                    $('#stockCount').text('Showing ' + filtered + ' stocks');
                    $('#totalStocks').text(total);
                    $('#visibleStocks').text(visible);
                }

                // Update pagination buttons
                function updatePagination() {
                    var info = table.page.info();
                    $('#currentPage').text(info.page + 1);

                    $('#prevPage').prop('disabled', info.page === 0);
                    $('#nextPage').prop('disabled', info.page + 1 === info.pages);
                }

                // Initial pagination update
                updatePagination();

                // Hide default DataTables elements
                $('.dataTables_length, .dataTables_filter, .dataTables_info, .dataTables_paginate').hide();
            });

            // Mini charts for Today Top Movers
            const miniCharts = [{
                    id: 'mini-chart-adbe',
                    trend: 'up',
                    color: '#10B981'
                },
                {
                    id: 'mini-chart-abnb',
                    trend: 'down',
                    color: '#EF4444'
                }
            ];

            miniCharts.forEach(chartConfig => {
                const ctx = document.getElementById(chartConfig.id).getContext('2d');
                const data = generateStockData(chartConfig.trend, 10, 3);
                createChart(ctx, data, chartConfig.color);
            });
        });
    </script>
</body>

</html>
