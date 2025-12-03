<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArbiterStock News</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .news-card {
            transition: all 0.3s ease;
        }
        
        .news-card:hover {
            transform: translateY(-8px);
        }
        
        .news-card:hover .news-image {
            transform: scale(1.1);
        }
        
        .news-image {
            transition: transform 0.5s ease;
        }
        
        .badge {
            backdrop-filter: blur(8px);
        }
        
        .skeleton {
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: loading 1.5s infinite;
        }
        
        @keyframes loading {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        .truncate-3 {
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
    </style>
</head>
<body class="bg-gray-50 text-gray-900">

    <x-navbar></x-navbar>

    <!-- HERO SECTION -->
    <div class="bg-gradient-to-r from-blue-700 to-purple-800 text-white py-20">
        <div class="max-w-7xl mx-auto px-6">
            <h1 class="text-5xl md:text-6xl font-extrabold mb-4">News</h1>
            <p class="text-blue-200 text-lg md:text-xl">Tetap terupdate dengan berita terbaru yang mempengaruhi harga saham dan tren pasar.</p>
        </div>
    </div>

    <!-- CONTENT -->
    <div class="max-w-7xl mx-auto py-12 px-6">

        <!-- Filter Tags -->
        <div class="flex gap-3 mb-8 flex-wrap">
            <button class="px-4 py-2 bg-blue-600 text-white rounded-full text-sm font-medium hover:bg-blue-700 transition">
                All News
            </button>
            <button class="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-full text-sm font-medium hover:bg-gray-50 transition">
                Markets
            </button>
            <button class="px-4 py-2 bg-white border border-gray-300 text-gray-700 rounded-full text-sm font-medium hover:bg-gray-50 transition">
                Technology
            </button>
        </div>

        <!-- Loading State -->
        <div id="loadingState" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            @for($i = 0; $i < 6; $i++)
            <div class="bg-white rounded-xl overflow-hidden shadow-md">
                <div class="skeleton h-56 w-full"></div>
                <div class="p-5">
                    <div class="skeleton h-4 w-3/4 mb-3 rounded"></div>
                    <div class="skeleton h-4 w-full mb-2 rounded"></div>
                    <div class="skeleton h-4 w-5/6 rounded"></div>
                </div>
            </div>
            @endfor
        </div>

        <!-- News Grid -->
        <div id="newsGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 hidden">
            <!-- Will be populated by JavaScript -->
        </div>

        <!-- Error State -->
        <div id="errorState" class="hidden text-center py-12">
            <svg class="w-16 h-16 mx-auto text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <h3 class="text-xl font-semibold text-gray-700 mb-2">Unable to load news</h3>
            <p class="text-gray-500 mb-4">Please try again later</p>
            <button onclick="loadNews()" class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
                Retry
            </button>
        </div>
    </div>

    <script>
        // Fetch news from Yahoo Finance RSS feed via API
        async function loadNews() {
            const loadingState = document.getElementById('loadingState');
            const newsGrid = document.getElementById('newsGrid');
            const errorState = document.getElementById('errorState');

            try {
                loadingState.classList.remove('hidden');
                newsGrid.classList.add('hidden');
                errorState.classList.add('hidden');

                // Using RSS2JSON service to convert Yahoo Finance RSS to JSON
                const response = await fetch('https://api.rss2json.com/v1/api.json?rss_url=https://finance.yahoo.com/news/rssindex');
                
                if (!response.ok) throw new Error('Failed to fetch news');
                
                const data = await response.json();
                
                if (data.status !== 'ok') throw new Error('Invalid response');

                const newsItems = data.items.slice(0, 12); // Get first 12 items

                newsGrid.innerHTML = newsItems.map(item => {
                    // Extract image from content or use default
                    let imageUrl = 'https://via.placeholder.com/800x450/4F46E5/FFFFFF?text=Financial+News';
                    
                    if (item.thumbnail) {
                        imageUrl = item.thumbnail;
                    } else if (item.enclosure && item.enclosure.link) {
                        imageUrl = item.enclosure.link;
                    }

                    // Format date
                    const date = new Date(item.pubDate);
                    const timeAgo = getTimeAgo(date);

                    // Extract category
                    const category = item.categories && item.categories.length > 0 
                        ? item.categories[0] 
                        : 'Finance';

                    // Clean description
                    const description = item.description
                        .replace(/<[^>]*>/g, '')
                        .substring(0, 150) + '...';

                    return `
                        <a href="${item.link}" target="_blank" class="news-card group block bg-white rounded-xl overflow-hidden shadow-md hover:shadow-2xl">
                            <div class="relative overflow-hidden h-56">
                                <img src="${imageUrl}" 
                                     class="news-image w-full h-full object-cover"
                                     onerror="this.src='https://via.placeholder.com/800x450/4F46E5/FFFFFF?text=Financial+News'">
                                
                                <div class="absolute top-4 left-4">
                                    <span class="badge bg-blue-600/90 text-white px-3 py-1 rounded-full text-xs font-semibold">
                                        ${category}
                                    </span>
                                </div>
                            </div>

                            <div class="p-5">
                                <h3 class="font-bold text-lg leading-tight mb-2 group-hover:text-blue-600 transition line-clamp-2">
                                    ${item.title}
                                </h3>
                                
                                <p class="text-gray-600 text-sm mb-3 truncate-3">
                                    ${description}
                                </p>

                                <div class="flex items-center justify-between text-xs text-gray-500">
                                    <span class="flex items-center gap-1">
                                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                        </svg>
                                        ${timeAgo}
                                    </span>
                                    <span class="text-blue-600 font-medium group-hover:underline">
                                        Read more →
                                    </span>
                                </div>
                            </div>
                        </a>
                    `;
                }).join('');

                loadingState.classList.add('hidden');
                newsGrid.classList.remove('hidden');

            } catch (error) {
                console.error('Error loading news:', error);
                loadingState.classList.add('hidden');
                errorState.classList.remove('hidden');
            }
        }

        function getTimeAgo(date) {
            const seconds = Math.floor((new Date() - date) / 1000);
            
            const intervals = {
                year: 31536000,
                month: 2592000,
                week: 604800,
                day: 86400,
                hour: 3600,
                minute: 60
            };

            for (const [unit, secondsInUnit] of Object.entries(intervals)) {
                const interval = Math.floor(seconds / secondsInUnit);
                if (interval >= 1) {
                    return interval === 1 ? `1 ${unit} ago` : `${interval} ${unit}s ago`;
                }
            }
            
            return 'Just now';
        }

        // Load news on page load
        document.addEventListener('DOMContentLoaded', loadNews);
    </script>

</body>
</html>