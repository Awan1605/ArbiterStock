<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ArbiterStock</title>

  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f5f7fa;
    }

    /* NAVBAR */
    nav {
      width: 100%;
      background: #000000ff;
      padding: 20px 40px;
      color: white;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-sizing: border-box;
    }

    nav .logo {
      font-size: 26px;
      font-weight: bold;
    }

    nav .menu {
      display: flex;
      gap: 40px;
    }

    nav .menu a {
      color: white;
      text-decoration: none;
      font-size: 18px;
      cursor: pointer;
    }

    nav .account-icon {
      width: 30px;
      height: 30px;
      border: 3px solid white;
      border-radius: 50%;
    }

    /* PAGE LAYOUT */
    .container {
      display: flex;
      padding: 30px;
      gap: 20px;
    }

    /* LEFT SECTION: TABLE */
    .market-section {
      flex: 2;
    }

    .search-box input {
      width: 100%;
      padding: 15px;
      border-radius: 12px;
      border: 1px solid #ccc;
      font-size: 16px;
      margin-bottom: 20px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 10px;
      overflow: hidden;
    }

    table th,
    table td {
      padding: 12px;
      border-bottom: 1px solid #eee;
      text-align: left;
      font-size: 14px;
    }

    table th {
      background: #f2f2f2;
    }

    /* RIGHT SECTION: NEWS */
    .news-section {
      flex: 1;
    }

    .news-title {
      font-size: 22px;
      font-weight: bold;
      margin-bottom: 20px;
      text-align: center;
    }

    .news-card {
      background: white;
      border-radius: 12px;
      padding: 15px;
      display: flex;
      gap: 15px;
      margin-bottom: 20px;
      align-items: center;
      box-shadow: 0 3px 7px rgba(0, 0, 0, 0.1);
    }

    .news-card img {
      width: 120px;
      height: 80px;
      border-radius: 10px;
      object-fit: cover;
    }
  </style>
</head>
<body>

  <!-- NAVBAR -->
  <nav>
    <div class="logo">ArbiterStock</div>

    <div class="menu">
      <a>Market</a>
      <a>News</a>
      <a>About Us</a>
    </div>

    <div class="account-icon"></div>
  </nav>

  <!-- MAIN CONTENT -->
  <div class="container">

    <!-- LEFT: TABLE -->
    <div class="market-section">
      <div class="search-box">
        <input type="text" placeholder="Search Market Here">
      </div>

      <table>
        <thead>
          <tr>
            <th>No</th>
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
          <tr><td>1</td><td>APL</td><td>$178.45K</td><td>0.82%</td><td>3.54%</td><td>$2.84T</td><td>$179.12</td><td>$176.90</td><td>$58.3B</td></tr>
          <tr><td>2</td><td>MSFT</td><td>$410.25K</td><td>1.25%</td><td>5.12%</td><td>$3.12T</td><td>$411.05</td><td>$408.73</td><td>$45.7B</td></tr>
          <tr><td>3</td><td>GOOGL</td><td>$142.17K</td><td>0.59%</td><td>2.18%</td><td>$1.89T</td><td>$143.05</td><td>$141.56</td><td>$32.4B</td></tr>
        </tbody>
      </table>
    </div>

    <!-- RIGHT: NEWS -->
    <div class="news-section">
      <div class="news-title">LATEST NEWS</div>

      <div class="news-card">
        <img src="https://via.placeholder.com/120x80">
        <p>Facebook’s ‘Failed’ Libra Cryptocurrency Is No Closer to Release</p>
      </div>

      <div class="news-card">
        <img src="https://via.placeholder.com/120x80">
        <p>The best bullish case ever made for Tesla</p>
      </div>

      <div class="news-card">
        <img src="https://via.placeholder.com/120x80">
        <p>Apple has soared out of the value realm...</p>
      </div>
    </div>

  </div>

</body>
</html>
