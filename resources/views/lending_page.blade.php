<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arbiterstocks - Platform Prediksi Harga Saham</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        
        .gradient-text {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .gradient-bg {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        }
        
        .card-hover {
            transition: all 0.3s ease;
        }
        
        .card-hover:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }
        
        .feature-icon {
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 16px;
            margin-bottom: 20px;
        }
        
        .hero-image {
            animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }
    </style>
</head>
<body class="bg-gray-50">
    
    <!-- Navigation -->
    <nav class="bg-white shadow-sm py-4">
        <div class="container mx-auto px-4">
            <div class="flex justify-between items-center">
                <!-- Logo -->
                <div class="flex items-center">
                    <a href="/" class="text-2xl font-bold">
                        <span class="text-gray-900">Arbiter</span>
                        <span class="gradient-text">stocks</span>
                    </a>
                </div>
                
                <!-- Desktop Menu -->
                <div class="hidden md:flex items-center space-x-8">
                    <a href="/lending_page" class="text-blue-600 font-semibold border-b-2 border-blue-600 pb-1">
                        <i class="fas fa-home mr-2"></i>Beranda
                    </a>
                    <a href="/market" class="text-gray-600 hover:text-blue-600 font-medium">
                        <i class="fas fa-chart-line mr-2"></i>Market
                    </a>
                    <a href="/news" class="text-gray-600 hover:text-blue-600 font-medium">
                        <i class="fas fa-newspaper mr-2"></i>News
                    </a>
                    <a href="/about" class="text-gray-600 hover:text-blue-600 font-medium">
                        <i class="fas fa-info-circle mr-2"></i>About Us
                    </a>
                    <a href="/contact" class="text-gray-600 hover:text-blue-600 font-medium">
                        <i class="fas fa-envelope mr-2"></i>Kontak
                    </a>
                </div>
                
                <!-- Right Side -->
                <div class="flex items-center space-x-4">
                    <a href="/login" class="text-blue-600 font-medium hover:text-blue-800">
                        Masuk
                    </a>
                    <a href="/register" class="gradient-bg text-white px-5 py-2 rounded-lg font-semibold hover:shadow-lg transition-shadow">
                        Daftar Gratis
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Hero Section -->
    <section class="py-16 md:py-24">
        <div class="container mx-auto px-4">
            <div class="flex flex-col lg:flex-row items-center">
                <!-- Left Content -->
                <div class="lg:w-1/2 mb-12 lg:mb-0 lg:pr-12">
                    <div class="inline-flex items-center px-4 py-2 rounded-full bg-blue-100 text-blue-600 font-medium mb-6">
                        <i class="fas fa-brain mr-2"></i>Arbiterstocks AI
                    </div>
                    <h1 class="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
                        Prediksi Harga Saham dengan 
                        <span class="gradient-text">Kecerdasan Buatan</span>
                    </h1>
                    <p class="text-lg text-gray-600 mb-8">
                        Arbiterstocks adalah platform analisis prediktif yang menggunakan teknologi machine learning 
                        untuk membantu Anda memahami pergerakan harga saham dengan lebih baik.
                    </p>
                    <div class="flex flex-col sm:flex-row gap-4">
                        <a href="/register" class="gradient-bg text-white px-8 py-3 rounded-lg font-semibold hover:shadow-xl transition-all">
                            <i class="fas fa-rocket mr-2"></i>Coba Gratis Sekarang
                        </a>
                        {{-- <a href="#features" class="bg-white text-blue-600 px-8 py-3 rounded-lg font-semibold border border-blue-600 hover:bg-blue-50 transition-all">
                            <i class="fas fa-play-circle mr-2"></i>Lihat Demo
                        </a> --}}
                    </div>
                    <div class="mt-8 flex items-center text-gray-500">
                        <i class="fas fa-check-circle text-green-500 mr-2"></i>
                        <span class="text-sm">Tidak perlu kartu kredit • 14 hari trial gratis</span>
                    </div>
                </div>
                
                <!-- Right Content -->
                <div class="lg:w-1/2">
                    <div class="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-8 hero-image">
                        <div class="bg-white rounded-xl shadow-lg p-6">
                            <div class="flex items-center mb-6">
                                <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mr-4">
                                    <i class="fas fa-chart-line text-blue-600 text-xl"></i>
                                </div>
                                <div>
                                    <h3 class="font-bold text-gray-900">Visualisasi Prediktif</h3>
                                    <p class="text-sm text-gray-600">Analisis pola harga masa depan</p>
                                </div>
                            </div>
                            <div class="space-y-4">
                                <div class="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                                    <span class="text-gray-700">Akurasi Prediksi</span>
                                    <span class="font-bold text-green-600">87.5%</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                                    <span class="text-gray-700">Saham Teranalisis</span>
                                    <span class="font-bold text-blue-600">500+</span>
                                </div>
                                <div class="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                                    <span class="text-gray-700">Update Data</span>
                                    <span class="font-bold text-purple-600">Real-time</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Features -->
    <section id="features" class="py-16 bg-white">
        <div class="container mx-auto px-4">
            <div class="text-center mb-16">
                <h2 class="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Fitur Utama Platform Kami</h2>
                <p class="text-lg text-gray-600 max-w-3xl mx-auto">
                    Teknologi canggih yang dirancang untuk memberikan insight prediktif terbaik
                </p>
            </div>
            
            <div class="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
                <!-- Feature 1 -->
                <div class="bg-gray-50 rounded-xl p-8 card-hover">
                    <div class="feature-icon gradient-bg">
                        <i class="fas fa-brain text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Machine Learning</h3>
                    <p class="text-gray-600">
                        Algoritma AI yang terus belajar dari data historis untuk meningkatkan akurasi prediksi.
                    </p>
                </div>
                
                <!-- Feature 2 -->
                <div class="bg-gray-50 rounded-xl p-8 card-hover">
                    <div class="feature-icon bg-green-500">
                        <i class="fas fa-chart-line text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Analisis Teknis</h3>
                    <p class="text-gray-600">
                        Indikator teknikal lengkap dan pola grafik untuk analisis mendalam.
                    </p>
                </div>
                
                <!-- Feature 3 -->
                <div class="bg-gray-50 rounded-xl p-8 card-hover">
                    <div class="feature-icon bg-purple-500">
                        <i class="fas fa-database text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Data Real-time</h3>
                    <p class="text-gray-600">
                        Akses data pasar saham terkini dengan update setiap menit.
                    </p>
                </div>
                
                <!-- Feature 4 -->
                <div class="bg-gray-50 rounded-xl p-8 card-hover">
                    <div class="feature-icon bg-orange-500">
                        <i class="fas fa-chart-bar text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Visualisasi Interaktif</h3>
                    <p class="text-gray-600">
                        Grafik interaktif untuk memahami trend dan pola pergerakan harga.
                    </p>
                </div>
                
                <!-- Feature 5 -->
                <div class="bg-gray-50 rounded-xl p-8 card-hover">
                    <div class="feature-icon bg-red-500">
                        <i class="fas fa-bell text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Notifikasi Cerdas</h3>
                    <p class="text-gray-600">
                        Alert otomatis ketika ada pola menarik atau peluang prediktif.
                    </p>
                </div>
                
                <!-- Feature 6 -->
                <div class="bg-gray-50 rounded-xl p-8 card-hover">
                    <div class="feature-icon bg-indigo-500">
                        <i class="fas fa-mobile-alt text-white text-2xl"></i>
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Akses Multi-Device</h3>
                    <p class="text-gray-600">
                        Akses platform dari desktop, tablet, atau smartphone Anda.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <section class="py-16 bg-gradient-to-r from-blue-50 to-indigo-50">
        <div class="container mx-auto px-4">
            <div class="text-center mb-16">
                <h2 class="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Bagaimana Cara Kerjanya?</h2>
                <p class="text-lg text-gray-600 max-w-3xl mx-auto">
                    Tiga langkah sederhana untuk mulai menggunakan platform prediksi kami
                </p>
            </div>
            
            <div class="grid md:grid-cols-3 gap-8">
                <!-- Step 1 -->
                <div class="text-center">
                    <div class="w-16 h-16 gradient-bg rounded-full flex items-center justify-center mx-auto mb-6 text-white text-2xl font-bold">
                        1
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Daftar Akun</h3>
                    <p class="text-gray-600">
                        Buat akun gratis Anda dalam 2 menit. Tidak perlu informasi kartu kredit.
                    </p>
                </div>
                
                <!-- Step 2 -->
                <div class="text-center">
                    <div class="w-16 h-16 gradient-bg rounded-full flex items-center justify-center mx-auto mb-6 text-white text-2xl font-bold">
                        2
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Pilih Saham</h3>
                    <p class="text-gray-600">
                        Pilih saham favorit Anda dari database kami yang mencakup 500+ perusahaan.
                    </p>
                </div>
                
                <!-- Step 3 -->
                <div class="text-center">
                    <div class="w-16 h-16 gradient-bg rounded-full flex items-center justify-center mx-auto mb-6 text-white text-2xl font-bold">
                        3
                    </div>
                    <h3 class="text-xl font-bold text-gray-900 mb-3">Analisis & Pelajari</h3>
                    <p class="text-gray-600">
                        Lihat prediksi, analisis, dan insight yang dihasilkan oleh sistem AI kami.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- Tolong update dan tambahkan untuk Fedbeck pengguna -->

    <!-- Testimonials -->
    <section class="py-16 bg-white">
        <div class="container mx-auto px-4">
            <div class="text-center mb-16">
                <h2 class="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Apa Kata Pengguna Kami</h2>
                <p class="text-lg text-gray-600 max-w-3xl mx-auto">
                    Ribuan investor telah menggunakan platform kami untuk analisis prediktif
                </p>
            </div>
            
            <div class="grid md:grid-cols-3 gap-8">
                <!-- Testimonial 1 -->
                <div class="bg-gray-50 rounded-xl p-8">
                    <div class="flex items-center mb-6">
                        <div class="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mr-4">
                            <i class="fas fa-user text-blue-600"></i>
                        </div>
                        <div>
                            <h4 class="font-bold text-gray-900">Budi Santoso</h4>
                            <p class="text-sm text-gray-600">Investor Saham, 5 tahun pengalaman</p>
                        </div>
                    </div>
                    <p class="text-gray-600 italic">
                        "Platform ini membantu saya memahami pola pasar dengan lebih baik. Prediksi yang diberikan cukup akurat untuk analisis jangka pendek."
                    </p>
                    <div class="flex text-yellow-400 mt-4">
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                    </div>
                </div>
                
                <!-- Testimonial 2 -->
                <div class="bg-gray-50 rounded-xl p-8">
                    <div class="flex items-center mb-6">
                        <div class="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mr-4">
                            <i class="fas fa-user text-green-600"></i>
                        </div>
                        <div>
                            <h4 class="font-bold text-gray-900">Sari Dewi</h4>
                            <p class="text-sm text-gray-600">Analis Keuangan</p>
                        </div>
                    </div>
                    <p class="text-gray-600 italic">
                        "Visualisasi data yang sangat interaktif dan mudah dipahami. Sangat membantu untuk presentasi analisis ke klien."
                    </p>
                    <div class="flex text-yellow-400 mt-4">
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star-half-alt"></i>
                    </div>
                </div>
                
                <!-- Testimonial 3 -->
                <div class="bg-gray-50 rounded-xl p-8">
                    <div class="flex items-center mb-6">
                        <div class="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mr-4">
                            <i class="fas fa-user text-purple-600"></i>
                        </div>
                        <div>
                            <h4 class="font-bold text-gray-900">Agus Wijaya</h4>
                            <p class="text-sm text-gray-600">Trader Pemula</p>
                        </div>
                    </div>
                    <p class="text-gray-600 italic">
                        "Sebagai pemula, platform ini sangat membantu. Interface yang user-friendly dan penjelasan yang mudah dimengerti."
                    </p>
                    <div class="flex text-yellow-400 mt-4">
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                        <i class="fas fa-star"></i>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- CTA -->
    <section class="py-16 gradient-bg">
        <div class="container mx-auto px-4">
            <div class="max-w-3xl mx-auto text-center text-white">
                <h2 class="text-3xl md:text-4xl font-bold mb-6">Siap Memulai Analisis Prediktif?</h2>
                <p class="text-xl mb-8 opacity-90">
                    Bergabunglah dengan ribuan investor yang telah menggunakan Arbiterstocks untuk analisis saham yang lebih baik.
                </p>
                <div class="flex flex-col sm:flex-row gap-4 justify-center">
                    <a href="/register" 
                       class="bg-white text-blue-600 px-8 py-4 rounded-lg font-bold text-lg hover:bg-blue-50 transition-colors">
                        <i class="fas fa-user-plus mr-2"></i>Daftar Gratis Sekarang
                    </a>
                    {{-- <a href="/demo" 
                       class="bg-transparent border-2 border-white text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-white/10 transition-colors">
                        <i class="fas fa-play-circle mr-2"></i>Lihat Demo Platform
                    </a> --}}
                </div>
                <p class="text-sm opacity-80 mt-6">
                    <i class="fas fa-shield-alt mr-2"></i>100% aman • Data terenkripsi • Tidak ada biaya tersembunyi
                </p>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-900 text-white py-12">
        <div class="container mx-auto px-4">
            <div class="grid md:grid-cols-4 gap-8 mb-8">
                <div>
                    <h3 class="text-2xl font-bold mb-4">
                        <span class="text-white">Arbiter</span>
                        <span class="text-blue-400">stocks</span>
                    </h3>
                    <p class="text-gray-400 mb-4">
                        Platform analisis prediktif harga saham berbasis kecerdasan buatan untuk investor Indonesia.
                    </p>
                    <div class="flex space-x-4">
                        <a href="#" class="text-gray-400 hover:text-white"><i class="fab fa-twitter"></i></a>
                        <a href="#" class="text-gray-400 hover:text-white"><i class="fab fa-linkedin"></i></a>
                        <a href="#" class="text-gray-400 hover:text-white"><i class="fab fa-instagram"></i></a>
                        <a href="#" class="text-gray-400 hover:text-white"><i class="fab fa-youtube"></i></a>
                    </div>
                </div>
                
                <div>
                    <h4 class="font-bold mb-4">Platform</h4>
                    <ul class="space-y-2">
                        <li><a href="/lending_page" class="text-gray-400 hover:text-white">Beranda</a></li>
                        <li><a href="/features" class="text-gray-400 hover:text-white">Fitur</a></li>
                        <li><a href="/pricing" class="text-gray-400 hover:text-white">Harga</a></li>
                        <li><a href="/demo" class="text-gray-400 hover:text-white">Demo</a></li>
                    </ul>
                </div>
                
                <div>
                    <h4 class="font-bold mb-4">Perusahaan</h4>
                    <ul class="space-y-2">
                        <li><a href="/about" class="text-gray-400 hover:text-white">Tentang Kami</a></li>
                        <li><a href="/blog" class="text-gray-400 hover:text-white">Blog</a></li>
                        <li><a href="/careers" class="text-gray-400 hover:text-white">Karir</a></li>
                        <li><a href="/contact" class="text-gray-400 hover:text-white">Kontak</a></li>
                    </ul>
                </div>
                
                <div>
                    <h4 class="font-bold mb-4">Legal</h4>
                    <ul class="space-y-2">
                        <li><a href="/privacy" class="text-gray-400 hover:text-white">Kebijakan Privasi</a></li>
                        <li><a href="/terms" class="text-gray-400 hover:text-white">Syarat & Ketentuan</a></li>
                        <li><a href="/disclaimer" class="text-gray-400 hover:text-white">Disclaimer</a></li>
                        <li><a href="/security" class="text-gray-400 hover:text-white">Keamanan</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="border-t border-gray-800 pt-8">
                <div class="flex flex-col md:flex-row justify-between items-center">
                    <div class="mb-4 md:mb-0">
                        <p class="text-gray-400 text-sm">
                            <i class="fas fa-map-marker-alt mr-2"></i>Batam, Indonesia
                        </p>
                    </div>
                    <div class="text-center md:text-right">
                        <p class="text-gray-500 text-sm">
                            © 2024 Arbiterstocks. Semua hak dilindungi undang-undang.
                        </p>
                        <p class="text-gray-500 text-xs mt-2">
                            Informasi yang disajikan hanya untuk tujuan analisis dan edukasi. Bukan merupakan rekomendasi investasi.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </footer>

    <script>
        // Simple animation for cards on scroll
        document.addEventListener('DOMContentLoaded', function() {
            const observerOptions = {
                threshold: 0.1
            };
            
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }
                });
            }, observerOptions);
            
            // Observe all feature cards
            document.querySelectorAll('.card-hover').forEach(card => {
                card.style.opacity = '0';
                card.style.transform = 'translateY(20px)';
                card.style.transition = 'all 0.5s ease';
                observer.observe(card);
            });
            
            // Set current year in footer
            document.getElementById('currentYear').textContent = new Date().getFullYear();
        });
    </script>
</body>
</html>