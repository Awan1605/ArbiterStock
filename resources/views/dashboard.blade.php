<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard | ArbiterStock</title>
    <link rel="stylesheet" href="{{ asset('css/dashboard.css') }}">
</head>
<body>
    <header>
        <h1>ArbiterStock Dashboard</h1>
        <nav>
            <a href="#market">Market</a>
            <a href="#news">News</a>
            <a href="#about">About Us</a>
        </nav>
    </header>

    <main>
        @include('market')
        @include('news')
        @include('about')
    </main>

    <footer>
        <p>&copy; 2025 ArbiterStock. All rights reserved.</p>
    </footer>
</body>
</html>
