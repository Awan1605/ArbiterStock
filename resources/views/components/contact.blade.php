<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <title>Hubungi Kami</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <script src="https://cdn.tailwindcss.com"></script>
</head>

<body class="bg-gradient-to-b from-blue-50 to-blue-100 min-h-screen">

    <div class="max-w-6xl mx-auto px-6 py-14">

        <!-- JUDUL -->
        <div class="text-center mb-12">
            <h1 class="text-4xl font-bold text-gray-800">Hubungi Kami</h1>
            <p class="text-gray-600 mt-3">
                Arbiterstock siap membantu Anda. Silakan hubungi kami melalui form di bawah ini.
            </p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8">

            <!-- INFORMASI KONTAK -->
            <div class="bg-white rounded-2xl shadow-lg p-8">

                <h2 class="text-xl font-bold mb-6">Informasi Kontak</h2>

                <div class="flex items-start gap-4 mb-6">
                    <div class="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">📞</div>
                    <div>
                        <p class="font-semibold">Telepon</p>
                        <p class="text-gray-600">+62 21 1234 5678</p>
                        <p class="text-gray-600">+62 812 3456 7890</p>
                    </div>
                </div>

                <div class="flex items-start gap-4 mb-6">
                    <div class="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">✉️</div>
                    <div>
                        <p class="font-semibold">Email</p>
                        <p class="text-gray-600">arbiterstock@gmail.com</p>
                        <p class="text-gray-600">muhammad21@gmail.com</p>
                    </div>
                </div>

                <div class="flex items-start gap-4 mb-8">
                    <div class="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center text-blue-600">📍</div>
                    <div>
                        <p class="font-semibold">Alamat</p>
                        <p class="text-gray-600">
                            Jl. Sudirman No. 123<br>
                            Batam, Kepulauan Riau<br>
                            Indonesia
                        </p>
                    </div>
                </div>

                <hr class="my-6">

                <h3 class="font-semibold mb-3"></h3>
                <div class="flex justify-between text-gray-600"><span>ArbiterStock hadir untuk menjembatani teknologi dan investasi. Kami percaya bahwa keputusan investasi yang baik lahir dari data yang kuat dan analisis yang objektif. Dengan AI sebagai inti, ArbiterStock berkomitmen menyediakan insight saham yang transparan, akurat, dan mudah diakses oleh semua investor.</span><span></span></div>
                <div class="flex justify-between text-gray-600 mt-1"><span></span><span></span></div>
                <div class="flex justify-between text-gray-600 mt-1"><span></span><span></span></div>
            </div>

            <!-- FORM KIRIM PESAN -->
            <div class="bg-white rounded-2xl shadow-lg p-8">

                <h2 class="text-xl font-bold mb-6">Kirim Pesan</h2>

                {{-- NOTIFIKASI SUKSES --}}
                @if(session('success'))
                    <div class="mb-4 bg-green-100 text-green-700 p-3 rounded-lg">
                        {{ session('success') }}
                    </div>
                @endif

                <form method="POST" action="{{ route('contact.store') }}">
                    @csrf

                    <div class="mb-4">
                        <label class="block mb-1 font-medium">Nama Lengkap</label>
                        <input type="text" name="name" required
                            class="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                            placeholder="Masukkan nama lengkap Anda">
                    </div>

                    <div class="mb-4">
                        <label class="block mb-1 font-medium">Email</label>
                        <input type="email" name="email" required
                            class="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                            placeholder="nama@email.com">
                    </div>

                    <div class="mb-4">
                        <label class="block mb-1 font-medium">Subjek</label>
                        <input type="text" name="subject" required
                            class="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                            placeholder="Subjek pesan Anda">
                    </div>

                    <div class="mb-6">
                        <label class="block mb-1 font-medium">Pesan</label>
                        <textarea rows="5" name="message" required
                            class="w-full border rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                            placeholder="Tulis pesan Anda di sini..."></textarea>
                    </div>

                    <button type="submit"
                        class="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-semibold">
                        ✈️ Kirim Pesan
                    </button>
                </form>
            </div>

        </div>
    </div>

</body>
</html>
