<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Login Page</title>

<style>
    body {
        margin: 0;
        font-family: Arial, sans-serif;
        background: #d7c9ff;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }

    .container {
        background: #e8dfff;
        width: 900px;
        padding: 40px;
        border-radius: 20px;
        display: flex;
        gap: 40px;
        align-items: center;
    }

    .left {
        width: 50%;
    }

    h1 {
        font-size: 40px;
        margin-bottom: 20px;
        font-weight: bold;
    }

    label {
        font-size: 18px;
        margin-top: 20px;
        display: block;
    }

    input {
        width: 100%;
        padding: 12px 15px;
        border-radius: 10px;
        border: none;
        margin-top: 10px;
        font-size: 16px;
    }

    .login-btn {
        width: 100%;
        padding: 15px;
        margin-top: 30px;
        border: none;
        background: #2f2f2f;
        color: white;
        font-size: 20px;
        border-radius: 10px;
        cursor: pointer;
    }

    .login-btn:hover {
        background: black;
    }

    .register-text {
        margin-top: 10px;
        font-size: 15px;
    }

    .register-text span {
        color: #0000ee;
        cursor: pointer;
        font-style: italic;
    }

    .right img {
        width: 350px;
    }
</style>

</head>
<body>

<div class="container">
    
    <!-- LEFT SIDE -->
    <div class="left">
        <h1>LOGIN</h1>

        <label>username</label>
        <input type="text" id="username" placeholder="input username">

        <label>password</label>
        <input type="password" id="password" placeholder="input password">

        <div class="register-text">
            don't have account? <span onclick="register()">register</span>
        </div>

        <button class="login-btn" onclick="login()">LOGIN</button>
    </div>

    <!-- RIGHT SIDE IMAGE -->
    <div class="right">
        <img src="https://i.imgur.com/0U5GtgP.png" alt="illustration">
    </div>

</div>

<script>
    function login() {
        let user = document.getElementById("username").value;
        let pass = document.getElementById("password").value;

        if (user === "" || pass === "") {
            alert("Username dan password tidak boleh kosong!");
        } else {
            alert("Login berhasil!\nUsername: " + user);
        }
    }

    function register() {
        alert("Menu register belum tersedia.\nSilakan buat halaman register sendiri 🙂");
    }
</script>

</body>
</html>
