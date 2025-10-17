<?php

namespace App\Http\Controllers\Auth;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;

class LoginController extends Controller
{
    public function showLoginForm()
    {
        // arahkan ke file resources/views/login.blade.php
        return view('login');
    }

    public function login(Request $request)
    {
        // sementara kosong dulu
        return "Login logic here";
    }
}
