<?php

use Illuminate\Support\Facades\Route;

// =======================
// Halaman DASHBOARD (default)
// =======================
Route::get('/', function () {
    return view('dashboard'); // arahkan ke resources/views/dashboard.blade.php
})->name('dashboard');


// =======================
// Halaman REGISTER
// =======================
Route::get('/register', function () {
    return view('auth.register');
})->name('register');

Route::post('/register', function () {
    // proses penyimpanan data register (nanti diganti dengan controller)
})->name('register.store');


// =======================
// Halaman LOGIN
// =======================
Route::get('/login', function () {
    return view('auth.login'); // arahkan ke file resources/views/auth/login.blade.php
})->name('login');

Route::post('/login', function () {
    // proses login (nanti diganti dengan controller)
})->name('login.process');
