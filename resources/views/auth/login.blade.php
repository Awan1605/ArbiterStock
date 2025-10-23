<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Login | ArbiterStock</title>
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
  </style>
</head>
<body>
  <div class="brand">ArbiterStock</div>
  <div class="container">
    <h1>LOGIN</h1>

    <form>
      <label for="username">username</label>
      <input type="text" id="username" placeholder="input username" />

      <label for="password">password</label>
      <input type="password" id="password" placeholder="input password" />

      <div class="small-text">
         don’t have account? <a href="{{ route('register') }}">register</a>
      </div>

      <button type="submit" class="btn">LOGIN</button>
    </form>
  </div>
</body>
</html>
