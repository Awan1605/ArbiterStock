<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register | ArbiterStock</title>
    <link rel="stylesheet" href="{{ asset('css/register.css') }}">
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="brand">ArbiterStock</div>

            <h2>REGISTER</h2>

            <form action="{{ route('register.store') }}" method="POST">
                @csrf
                <label for="username">username</label>
                <input type="text" name="username" id="username" placeholder="input username" required>

                <label for="email">email</label>
                <input type="email" name="email" id="email" placeholder="input email" required>

                <label for="password">password</label>
                <input type="password" name="password" id="password" placeholder="input password" required>

                <p class="login-text">already have account? <a href="{{ route('login') }}">login</a></p>

                <button type="submit">Register</button>
            </form>
        </div>
    </div>
</body>
</html>
