<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign In - ArbiterStock</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>

<body class="bg-gray-100 flex items-center justify-center min-h-screen">

    <div class="bg-white p-10 rounded-lg shadow-lg w-full max-w-md">

        <!-- Logo & Title -->
        <h1 class="text-3xl font-semibold text-center mb-2">ArbiterStock</h1>
        <p class="text-center text-gray-500 mb-8">
            AI-based stock price analysis application
        </p>

        <!-- Sign In Title -->
        <h2 class="text-2xl font-bold text-center mb-6">Sign In</h2>

        <!-- FORM LOGIN -->
        <form action="proses_login.php" method="POST" class="space-y-5">

            <div>
                <label class="block text-gray-600 mb-1">Email</label>
                <input type="email" name="email"
                       class="w-full border border-gray-300 px-4 py-3 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                       placeholder="input email" required>
            </div>

            <div>
                <label class="block text-gray-600 mb-1">Password</label>
                <input type="password" name="password"
                       class="w-full border border-gray-300 px-4 py-3 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
                       placeholder="Input password" required>
            </div>

            <button type="submit" a href="/market"
                    class="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 rounded-lg font-semibold text-lg">
                Sign In
            </button>

        </form>

        <!-- Forgot Password -->
        <div class="text-center mt-4">
            <a href="forgot_password.php" class="text-gray-600 hover:text-gray-800 text-sm">
                Forgot password?
            </a>
        </div>

        <!-- Sign Up Link -->
        <p class="text-center mt-5 text-sm">
            Don't have an account?
            <a href="register.blade.php" class="text-blue-600 font-semibold hover:underline">Sign Up</a>
        </p>

    </div>

</body>
</html>