<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - ArbiterStock</title>
    <style>
        body {
            font-family: "Poppins", sans-serif;
            background-color: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }

        h1 {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 30px;
            text-align: center;
            color: #222;
        }

        .logo {
            position: absolute;
            top: 25px;
            left: 40px;
            font-size: 20px;
            font-weight: 600;
            color: #222;
        }

        form {
            width: 320px;
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 14px;
            margin-bottom: 5px;
            color: #333;
        }

        input {
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 6px;
            margin-bottom: 18px;
            outline: none;
            transition: border-color 0.3s;
            font-size: 14px;
        }

        input:focus {
            border-color: #000;
        }

        button {
            padding: 14px;
            background-color: #222;
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #333;
        }

        .register {
            text-align: center;
            font-size: 13px;
            margin-top: 10px;
            color: #333;
        }

        .register a {
            color: #000;
            font-style: italic;
            text-decoration: none;
        }

        .register a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="logo">ArbiterStock</div>

    <h1>LOGIN</h1>

    <form action="{{ url('/login') }}" method="POST">
        @csrf
        <label for="username">username</label>
        <input type="text" id="username" name="username" placeholder="input username" required>

        <label for="password">password</label>
        <input type="password" id="password" name="password" placeholder="input password" required>

        <div class="register">
            don't have account? <a href="#">register</a>
        </div>

        <button type="submit">LOGIN</button>
    </form>
</body>
</html>
