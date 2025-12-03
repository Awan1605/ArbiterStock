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
</head>

<!-- Navbar -->
<nav class="bg-black shadow-sm py-4 sticky top-0 z-10">
    <div class="container mx-auto px-4 flex justify-between items-center">
        <div class="flex items-center space-x-10">
            <a href="/" class="text-xl font-bold">
                <span class="text-white">Arbiter</span><span class="text-blue-500">stocks</span>
            </a>
            <div class="hidden md:flex space-x-8">
                <a href="/"
                    class="text-gray-100 hover:text-blue-500 font-medium hover:underline underline-offset-4">
                    Beranda
                </a>
                <a href="/market"
                    class="text-gray-100 hover:text-blue-500 font-medium hover:underline underline-offset-4">
                    Markets
                </a>
                <a href="/news"
                    class="text-gray-100 hover:text-blue-500 font-medium hover:underline underline-offset-4">
                    News
                </a>
                <a href="/about"
                    class="text-gray-100 hover:text-blue-500 font-medium hover:underline underline-offset-4">
                    About Us
                </a>
            </div>
        </div>
        <div class="flex items-center space-x-4">
            <button class="p-2 rounded-full hover:bg-blue-900 transition-colors">
                <i class="fas fa-bell text-blue-400"></i>
            </button>
            <a href="/profile" class="p-2 rounded-full hover:bg-blue-900 transition-colors">
                <i class="fas fa-user text-blue-400"></i>
            </a>
            <a href="/get-started"
                class="bg-red-600 text-white px-2 py-2 rounded-full font-medium hover:bg-red-500 transition-colors">
                Logout
            </a>
        </div>
    </div>
</nav>