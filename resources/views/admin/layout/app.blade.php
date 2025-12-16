<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Admin | ArbiterStock</title>

    {{-- CSS ADMIN --}}
    <link rel="stylesheet" href="{{ asset('css/admin.css') }}">
</head>
<body>

<div class="admin">

    <!-- SIDEBAR -->
    <aside class="sidebar">
        <h2>ArbiterStock</h2>

        <div class="menu">
            <a href="/admin/dashboard" class="{{ request()->is('admin/dashboard') ? 'active' : '' }}">
                📊 Dashboard
            </a>

            <a href="/admin/user" class="{{ request()->is('admin/user') ? 'active' : '' }}">
                👤 Manajemen Pengguna
            </a>

            <a href="/admin/saham" class="{{ request()->is('admin/saham') ? 'active' : '' }}">
                📈 Data Saham
            </a>

            <a href="/admin/insight" class="{{ request()->is('admin/insight') ? 'active' : '' }}">
                🤖 Insight AI
            </a>
        </div>
    </aside>

    <!-- MAIN CONTENT -->
    <main class="main">
        @yield('content')
    </main>

</div>

</body>
</html>
