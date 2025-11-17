<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <script src="https://cdn.tailwindcss.com"></script>
  <title>ArbiterStock</title>
</head>

<body class="bg-gray-100">

  <!-- NAVBAR -->
  <nav class="bg-sky-500 text-white px-8 py-4 flex justify-between items-center">
    <h1 class="text-2xl font-semibold">ArbiterStock</h1>

    <ul class="flex gap-10 text-lg">
      <li><a href="#" class="hover:underline">Market</a></li>
      <li><a href="#" class="hover:underline">News</a></li>
      <li><a href="#" class="hover:underline">About Us</a></li>
    </ul>

    <!-- User Icon -->
    <div class="w-8 h-8 border border-white rounded-full flex items-center justify-center">
      <span class="text-xl">👤</span>
    </div>
  </nav>

  <!-- SEARCH BOX -->
  <div class="px-10 mt-8">
    <div class="bg-white shadow-md w-80 flex items-center gap-3 px-4 py-2 rounded-lg border">
      <input type="text" placeholder="Search Market Here"
        class="w-full outline-none text-gray-700" />
      <span class="text-gray-400 text-lg">🔍</span>
    </div>
  </div>

  <!-- TABLE -->
  <div class="px-10 mt-6">
    <div class="bg-white shadow-lg rounded-lg overflow-hidden">
      <table class="min-w-full text-left text-gray-700">
        <thead class="bg-gray-100">
          <tr>
            <th class="py-3 px-4">NO</th>
            <th class="py-3 px-4">Name Stock</th>
            <th class="py-3 px-4">Current Price</th>
            <th class="py-3 px-4">24h</th>
            <th class="py-3 px-4">7d</th>
            <th class="py-3 px-4">Market Cap</th>
            <th class="py-3 px-4">High</th>
            <th class="py-3 px-4">Low</th>
            <th class="py-3 px-4">Volume</th>
          </tr>
        </thead>

        <tbody>

          <!-- 1 -->
          <tr class="border-t">
            <td class="py-4 px-4">1</td>
            <td class="py-4 px-4 flex items-center gap-2">
              <img src="https://logo.clearbit.com/apple.com" class="w-6 h-6">
              AAPL (Apple Inc.)
            </td>
            <td class="py-4 px-4">$178.45k</td>
            <td class="py-4 px-4 flex items-center gap-1">📉 0.82%</td>
            <td class="py-4 px-4 flex items-center gap-1">📉 3.54%</td>
            <td class="py-4 px-4">$2.84T</td>
            <td class="py-4 px-4">$179.12k</td>
            <td class="py-4 px-4">$176.90k</td>
            <td class="py-4 px-4">$58.3B</td>
          </tr>

          <!-- 2 -->
          <tr class="border-t">
            <td class="py-4 px-4">2</td>
            <td class="py-4 px-4 flex items-center gap-2">
              <img src="https://logo.clearbit.com/microsoft.com" class="w-6 h-6">
              MSFT (Microsoft Corp.)
            </td>
            <td class="py-4 px-4">$410.25k</td>
            <td class="py-4 px-4">📉 1.25%</td>
            <td class="py-4 px-4">📉 5.12%</td>
            <td class="py-4 px-4">$3.12T</td>
            <td class="py-4 px-4">$411.05k</td>
            <td class="py-4 px-4">$406.73k</td>
            <td class="py-4 px-4">$45.7B</td>
          </tr>

          <!-- 3 -->
          <tr class="border-t">
            <td class="py-4 px-4">3</td>
            <td class="py-4 px-4 flex items-center gap-2">
              <img src="https://logo.clearbit.com/abc.xyz" class="w-6 h-6">
              GOOGL (Alphabet Inc.)
            </td>
            <td class="py-4 px-4">$142.17k</td>
            <td class="py-4 px-4">📉 0.59%</td>
            <td class="py-4 px-4">📉 2.18%</td>
            <td class="py-4 px-4">$1.89T</td>
            <td class="py-4 px-4">$143.05k</td>
            <td class="py-4 px-4">$141.56k</td>
            <td class="py-4 px-4">$32.4B</td>
          </tr>

          <!-- 4 -->
          <tr class="border-t">
            <td class="py-4 px-4">4</td>
            <td class="py-4 px-4 flex items-center gap-2">
              <img src="https://logo.clearbit.com/amazon.com" class="w-6 h-6">
              AMZN (Amazon.com Inc.)
            </td>
            <td class="py-4 px-4">$128.94k</td>
            <td class="py-4 px-4">📉 1.47%</td>
            <td class="py-4 px-4">📉 6.45%</td>
            <td class="py-4 px-4">$1.67T</td>
            <td class="py-4 px-4">$129.35k</td>
            <td class="py-4 px-4">$127.76k</td>
            <td class="py-4 px-4">$51.2B</td>
          </tr>

          <!-- 5 -->
          <tr class="border-t">
            <td class="py-4 px-4">5</td>
            <td class="py-4 px-4 flex items-center gap-2">
              <img src="https://logo.clearbit.com/tesla.com" class="w-6 h-6">
              TSLA (Tesla Inc.)
            </td>
            <td class="py-4 px-4">$256.78k</td>
            <td class="py-4 px-4">📉 0.94%</td>
            <td class="py-4 px-4">📉 4.22%</td>
            <td class="py-4 px-4">$812.5B</td>
            <td class="py-4 px-4">$259.23k</td>
            <td class="py-4 px-4">$254.56k</td>
            <td class="py-4 px-4">$72.8B</td>
          </tr>

        </tbody>
      </table>
    </div>
  </div>

</body>
</html>
