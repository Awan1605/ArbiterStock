<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArbiterStock News</title>

   
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-white text-gray-900">

    
    <nav class="bg-black text-white">

        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <h1 class="text-2xl font-semibold">ArbiterStock</h1>

            <ul class="flex gap-6">
                <li><a href="#" class="hover:text-gray-300">Market</a></li>
                <li><a href="#" class="hover:text-gray-300">News</a></li>
                <li><a href="#" class="hover:text-gray-300">About Us</a></li>
            </ul>

            <div class="w-7 h-7 rounded-full border border-white flex items-center justify-center">
                👤
            </div>
        </div>
    </nav>

    <!-- CONTENT -->
    <div class="max-w-7xl mx-auto py-10 px-6">

        <h2 class="text-3xl font-bold mb-6">News</h2>

        <!-- GRID -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

            @foreach([
                [
                    'slug' => 'biden-primary-calendar',
                    'title' => "Democrats Overhaul Party’s Primary Calendar, Upending a Political Tradition",
                    'image' => 'https://picsum.photos/800/450?random=1',
                    'author' => 'Ginny Dennis',
                    'time' => 'Just now'
                ],
                [
                    'slug' => 'fec-probes-santos',
                    'title' => "Justice Department asks FEC to stand down as prosecutors probe Santos",
                    'image' => 'https://picsum.photos/800/450?random=2',
                    'author' => 'Ginny Dennis',
                    'time' => 'Feb 4, 2023'
                ],
                [
                    'slug' => 'pakistan-leader-dies',
                    'title' => "Pervez Musharraf, Former Military Ruler of Pakistan, Dies at 79",
                    'image' => 'https://picsum.photos/800/450?random=3',
                    'author' => 'Ginny Dennis',
                    'time' => 'Feb 4, 2023'
                ],
                [
                    'slug' => 'abortion-decision',
                    'title' => "Fears mount around ‘catastrophic’ abortion pills case as decision nears",
                    'image' => 'https://picsum.photos/800/450?random=4',
                    'author' => 'Ginny Dennis',
                    'time' => 'Feb 4, 2023'
                ],
                [
                    'slug' => 'ukraine-president',
                    'title' => "U.S. Secretary of State speaks at a diplomatic conference",
                    'image' => 'https://picsum.photos/800/450?random=5',
                    'author' => 'Ginny Dennis',
                    'time' => 'Feb 4, 2023'
                ]
            ] as $item)
            
            <a href="/news/{{ $item['slug'] }}" class="group block relative rounded-lg overflow-hidden shadow-lg">
                <img src="{{ $item['image'] }}" class="w-full h-56 object-cover transition duration-300 group-hover:scale-105">

                <div class="absolute bottom-0 left-0 w-full p-4 bg-gradient-to-t from-black to-transparent text-white">
                    <h3 class="font-semibold leading-tight text-lg">{{ $item['title'] }}</h3>
                    <p class="text-xs opacity-70 mt-1">By {{ $item['author'] }} · {{ $item['time'] }}</p>
                </div>
            </a>

            @endforeach

        </div>
    </div>

</body>
</html>
