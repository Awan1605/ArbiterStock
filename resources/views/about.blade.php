<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About Us - ArbiterStock</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://kit.fontawesome.com/3c90f9f2a5.js" crossorigin="anonymous"></script>
</head>

<body class="bg-gray-100">
    <x-navbar></x-navbar>

    <!-- HERO SECTION -->
    <section class="max-w-6xl mx-auto px-6 py-10">

        <h2 class="text-3xl font-bold text-center mb-6">
            Kelompok 3 • IF D AI Malam
        </h2>

        <!-- FOTO KELOMPOK -->
        <div class="w-full flex justify-center mb-6">
            <img src="{{ asset('image/foto1.jpg') }}" 
                 alt="Team Photo" class="rounded-lg shadow-lg w-full max-w-4xl">
        </div>

        <!-- PENJELASAN -->
        <p class="text-center text-gray-700 leading-relaxed max-w-4xl mx-auto">
            Kelompok 3 IF D Malam mempersembahkan sebuah aplikasi prediksi harga saham 
            yang dibangun untuk mendukung pengguna dalam mengambil keputusan investasi 
            dengan lebih percaya diri dan terarah.
            <br><br>
            Berbekal analisis data mendalam dan model machine learning modern, aplikasi ini 
            mampu menyajikan prediksi yang tidak hanya akurat dan informatif, tetapi juga mudah 
            dipahami oleh siapa pun—mulai dari pemula hingga investor berpengalaman.
        </p>
    </section>

    <!-- TEAM SECTION -->
    <section class="max-w-6xl mx-auto px-6 pb-20">
        <h3 class="text-center text-xl font-semibold mb-10">Meet the Team</h3>

        @php
            $team = [
                ["nama" => "Nama 1", "role" => "Frontend Developer", "foto" => "foto1.jpg"],
                ["nama" => "Nama 2", "role" => "Backend Developer", "foto" => "foto2.jpg"],
                ["nama" => "Nama 3", "role" => "AI Engineer", "foto" => "foto3.jpg"],
                ["nama" => "Nama 4", "role" => "Fullstack Developer", "foto" => "foto4.jpg"],
                ["nama" => "Nama 5", "role" => "UI/UX Designer", "foto" => "foto5.jpg"],
                ["nama" => "Nama 6", "role" => "Project Manager", "foto" => "foto6.jpg"],
            ];
        @endphp

        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-10">

            @foreach($team as $t)
                <div class="bg-white rounded-lg shadow-md p-5 text-center">

                    <img src="{{ asset('image/foto1/' . $t['foto']) }}" 
                         class="w-32 h-32 object-cover mx-auto rounded-full shadow mb-4">

                    <h4 class="font-semibold text-lg">{{ $t['nama'] }}</h4>
                    <p class="text-gray-500 mb-3">{{ $t['role'] }}</p>

                    <div class="flex justify-center gap-3 text-blue-600">
                        <i class="fa-brands fa-instagram"></i>
                        <i class="fa-brands fa-facebook"></i>
                        <i class="fa-brands fa-twitter"></i>
                    </div>
                </div>
            @endforeach

        </div>
    </section>

</body>
</html>
