<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;

class AuthController extends Controller
{
    /**
     * Menampilkan form login
     */
    public function showLoginForm()
    {
        return view('auth.login'); // Pastikan file view login.blade.php ada di resources/views/auth/
    }

    /**
     * Memproses login
     */
    public function login(Request $request)
    {
        // Validasi input
        $credentials = $request->validate([
            'email' => 'required|email',
            'password' => 'required|min:6',
        ]);

        // Coba login
        if (Auth::attempt($credentials)) {
            // Regenerate session untuk mencegah session fixation
            $request->session()->regenerate();
            
            // Redirect ke halaman yang dimaksud atau dashboard
            return redirect()->intended('/dashboard');
        }

        // Jika login gagal, kembali ke form login dengan error
        return back()->withErrors([
            'email' => 'Email atau password yang dimasukkan salah.',
        ])->withInput($request->except('password'));
    }

    /**
     * Logout user
     */
    public function logout(Request $request)
    {
        Auth::logout();
        
        $request->session()->invalidate();
        $request->session()->regenerateToken();
        
        return redirect('/');
    }
}