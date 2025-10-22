<?php

use Illuminate\Support\Facades\Route;

// Halaman Register
Route::get('/register', function () {
    return view('auth.register');
})->name('register');

// Proses form Register (nanti bisa diisi controller)
Route::post('/register', function () {
    // proses penyimpanan data register (belum diisi)
})->name('register.store');

// Halaman Login (sementara placeholder)
Route::get('/login', function () {
    return 'Halaman login (buat nanti)';
})->name('login');
