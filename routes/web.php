<?php

use Illuminate\Support\Facades\Route;

// =======================
// Halaman Dashboard (utama)
// =======================
Route::get('/', function () {
    return view('lending_page');
})->name('lending_page');

// =======================
// Halaman Modular (opsional, bisa diakses langsung)
// =======================
Route::get('/market', function () {
    return view('market');
})->name('market');

Route::get('/news', function () {
    return view('news');
})->name('news');

Route::get('/about', function () {
    return view('about');
})->name('about');

// =======================
// Halaman AUTH: REGISTER
// =======================
Route::get('/register', function () {
    return view('auth.register'); // buat file register.blade.php di resources/views/auth/
})->name('register');

Route::post('/register', function () {
    // nanti bisa diisi logika penyimpanan user
})->name('register.store');

// =======================
// Halaman AUTH: LOGIN
// =======================
Route::get('/login', function () {
    return view('auth.login'); // buat file login.blade.php di resources/views/auth/
})->name('login');

Route::post('/login', function () {
    // nanti bisa diisi logika autentikasi
})->name('login.store');

// =======================
// Logout (opsional)
// =======================
Route::post('/logout', function () {
    // nanti bisa diisi logika logout
})->name('logout');
