<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dashboard | ArbiterStock</title>
  <link rel="stylesheet" href="{{ asset('css/dashboard.css') }}">
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    /* Tambahan CSS untuk tab aktif */
    .nav-links a {
      color: #333;
      text-decoration: none;
      margin: 0 15px;
      font-weight: 500;
      transition: color 0.3s;
      cursor: pointer;
    }
    .nav-links a:hover, .nav-links a.active {
      color: #007bff;
      border-bottom: 2px solid #007bff;
    }

    /* Sembunyikan section selain yang aktif */
    .section {
      display: none;
    }
    .section.active {
      display: block;
    }
  </style>
</head>
<body>
  <!-- 🔹 NAVBAR -->
  <header class="navbar">
    <div class="logo">ArbiterStock</div>
    <nav class="nav-links">
      <a class="active" data-target="market">Market</a>
      <a data-target="news">News</a>
      <a data-target="about">About Us</a>
    </nav>
    <div class="user-icon">👤</div>
  </header>

  <!-- 🔹 ISI DASHBOARD -->
  <main class="content">
    <!-- SECTION: MARKET -->
    <section id="market" class="section active">
      <div class="search-box">
        <input type="text" placeholder="Search Market Here">
      </div>

      <table class="market-table">
        <thead>
          <tr>
            <th>NO</th>
            <th>Name Stock</th>
            <th>Current Price</th>
            <th>24h</th>
            <th>7d</th>
            <th>Market Cap</th>
            <th>High</th>
            <th>Low</th>
            <th>Volume</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td><img src="https://logo.clearbit.com/apple.com" class="logo-icon"> AAPL (Apple Inc.)</td>
            <td>$178.45k</td><td>🔼0.82%</td><td>🔼3.54%</td><td>$2.84T</td><td>$179.12k</td><td>$176.90k</td><td>$58.3B</td>
          </tr>
          <tr>
            <td>2</td>
            <td><img src="https://logo.clearbit.com/microsoft.com" class="logo-icon"> MSFT (Microsoft Corp.)</td>
            <td>$410.25k</td><td>🔼1.25%</td><td>🔼5.12%</td><td>$3.12T</td><td>$411.05k</td><td>$406.73k</td><td>$45.7B</td>
          </tr>
          <tr>
            <td>3</td>
            <td><img src="https://logo.clearbit.com/google.com" class="logo-icon"> GOOGL (Alphabet Inc.)</td>
            <td>$142.17k</td><td>🔼0.59%</td><td>🔼2.18%</td><td>$1.89T</td><td>$143.05k</td><td>$141.56k</td><td>$32.4B</td>
          </tr>
        </tbody>
      </table>
    </section>

    <!-- SECTION: NEWS -->
    <section id="news" class="section">
      <h2>LATEST NEWS</h2>
      <div class="news-card">
        <img src="https://i.ibb.co/mDq6jwn/libra.jpg" alt="News 1">
        <p>Facebook's 'Failed' Libra Cryptocurrency Is No Closer to Release</p>
      </div>
      <div class="news-card">
        <img src="https://i.ibb.co/f4JKT7Y/tesla.jpg" alt="News 2">
        <p>The best bullish case ever made for Tesla, according to prominent Tesla bear</p>
      </div>
      <div class="news-card">
        <img src="https://i.ibb.co/BzQd6L7/applewatch.jpg" alt="News 3">
        <p>Apple has soared out of the value realm, but you may still be able to find success on this stock list</p>
      </div>
    </section>

    <!-- SECTION: ABOUT US -->
    <section id="about" class="section">
      <h2>About ArbiterStock</h2>
      <p>
        ArbiterStock adalah platform pemantauan pasar saham yang menyediakan data real-time dan berita terbaru untuk membantu investor membuat keputusan yang lebih baik.
      </p>
      <p>
        Kami berfokus pada transparansi, kecepatan, dan kemudahan akses informasi saham dari berbagai perusahaan teknologi besar di dunia.
      </p>
      <p>
        Tim kami terdiri dari analis keuangan dan pengembang teknologi yang berkomitmen menghadirkan pengalaman terbaik bagi pengguna kami.
      </p>
    </section>
  </main>

  <!-- 🔹 SCRIPT UNTUK GANTI TAB -->
  <script>
    const links = document.querySelectorAll('.nav-links a');
    const sections = document.querySelectorAll('.section');

    links.forEach(link => {
      link.addEventListener('click', () => {
        // Hapus kelas aktif dari semua link
        links.forEach(l => l.classList.remove('active'));
        // Tambahkan ke link yang diklik
        link.classList.add('active');

        // Sembunyikan semua section
        sections.forEach(s => s.classList.remove('active'));
        // Tampilkan section sesuai target
        const target = document.getElementById(link.dataset.target);
        target.classList.add('active');
      });
    });
  </script>
</body>
</html>
