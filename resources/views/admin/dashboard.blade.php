@extends('admin.layout.app')

@section('content')

<!-- TOPBAR -->
<div class="topbar">
    <div>
        <h2>Hello, Muhammad Risky 👋</h2>
        <small>Have a nice day</small>
    </div>

    <div class="profile">
        <span>🔔</span>
        <img src="https://i.pravatar.cc/100?img=12" alt="Admin">
        <div>
            <strong>Muhammad Risky</strong><br>
            <small>Admin</small>
        </div>
    </div>
</div>

<!-- CARDS -->
<div class="cards">
    <div class="card">
        <div class="icon blue">👥</div>
        <div>
            <p>Total User</p>
            <h3>40,689</h3>
            <span class="up">▲ 8.5% Up from yesterday</span>
        </div>
    </div>

    <div class="card">
        <div class="icon green">📈</div>
        <div>
            <p>Total Pengunjung</p>
            <h3>2,000</h3>
            <span class="down">▼ 4.3% Down from yesterday</span>
        </div>
    </div>

    <div class="card">
        <div class="icon yellow">📦</div>
        <div>
            <p>Total Feedback</p>
            <h3>100</h3>
            <span class="up">▲ 5% Up from yesterday</span>
        </div>
    </div>
</div>

<!-- CHART -->
<div class="chart">
    <div class="chart-head">
        <h3>Total Pengunjung</h3>
        <select>
            <option>October</option>
            <option>November</option>
        </select>
    </div>

    <canvas id="visitorChart" height="120"></canvas>
</div>

<!-- CHART JS -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
const ctx = document.getElementById('visitorChart').getContext('2d');

new Chart(ctx, {
    type: 'line',
    data: {
        labels: ['1', '5', '10', '15', '20', '25', '30'],
        datasets: [{
            data: [200, 450, 380, 700, 600, 550, 680],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59,130,246,0.15)',
            fill: true,
            tension: 0.4,
            pointRadius: 4,
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: { color: '#f1f5f9' }
            },
            x: {
                grid: { display: false }
            }
        }
    }
});
</script>

@endsection
