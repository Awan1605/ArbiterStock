<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Register | ArbiterStock</title>
  <style>
    body {
      font-family: "Poppins", sans-serif;
      background-color: #fff;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      margin: 0;
    }

    .container {
      text-align: center;
    }

    h1 {
      font-size: 2rem;
      margin-bottom: 1rem;
    }

    label {
      display: block;
      text-align: left;
      font-size: 0.9rem;
      margin-bottom: 0.3rem;
    }

    input {
      width: 250px;
      padding: 10px;
      border: 1px solid #ccc;
      border-radius: 6px;
      margin-bottom: 1rem;
      font-size: 0.9rem;
    }

    .small-text {
      font-size: 0.8rem;
      color: #444;
      margin-bottom: 1.5rem;
    }

    .small-text a {
      font-style: italic;
      color: #000;
      text-decoration: none;
    }

    .btn {
      width: 270px;
      padding: 12px;
      background-color: #222;
      color: white;
      font-weight: bold;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      transition: 0.2s ease;
    }

    .btn:hover {
      background-color: #444;
    }

    .brand {
      position: absolute;
      top: 20px;
      left: 40px;
      font-size: 1.3rem;
      font-weight: 500;
    }

    a.brand-link {
      text-decoration: none;
      color: black;
    }
  </style>
</head>
<body>
  <a href="/" class="brand-link"><div class="brand">ArbiterStock</div></a>

  <div class="container">
    <h1>REGISTER</h1>

    <form action="{{ route('register.store') }}" method="POST">
      @csrf
      <label for="username">username</label>
      <input type="text" id="username" name="username" placeholder="input username" required>

      <label for="email">email</label>
      <input type="email" id="email" name="email" placeholder="input email" required>

      <label for="password">password</label>
      <input type="password" id="password" name="password" placeholder="input password" required>

      <div class="small-text">
        already have account? <a href="{{ route('login') }}">login</a>
      </div>

      <button type="submit" class="btn">Register</button>
    </form>
  </div>
</body>
</html>
