<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Register - ArbiterStock</title>
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: #fff;
        }
        .container {
            text-align: center;
        }
        h2 {
            font-weight: 600;
            margin-bottom: 20px;
            letter-spacing: 1px;
        }
        label {
            display: block;
            text-align: left;
            font-size: 13px;
            margin-bottom: 5px;
            color: #333;
        }
        input {
            width: 300px;
            padding: 10px;
            margin-bottom: 15px;
            border: 1px solid #ccc;
            border-radius: 6px;
            font-size: 14px;
        }
        button {
            width: 320px;
            padding: 12px;
            background-color: #000;
            color: white;
            border: none;
            border-radius: 6px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }
        button:hover {
            background-color: #222;
        }
        .bottom-text {
            font-size: 13px;
            margin-top: 5px;
            color: #555;
        }
        .bottom-text a {
            font-style: italic;
            color: black;
            text-decoration: none;
        }
        .brand {
            position: absolute;
            top: 20px;
            left: 40px;
            font-weight: bold;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="brand">ArbiterStock</div>

    <div class="container">
        <h2>REGISTER</h2>
        <form action="{{ route('register') }}" method="POST">
            @csrf
            <label>username</label>
            <input type="text" name="username" placeholder="input username" required>

            <label>email</label>
            <input type="email" name="email" placeholder="input email" required>

            <label>password</label>
            <input type="password" name="password" placeholder="input password" required>

            <div class="bottom-text">
                already have account? <a href="{{ route('login') }}">login</a>
            </div>

            <button type="submit">Register</button>
        </form>
    </div>
</body>
</html>
