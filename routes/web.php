<?php

use App\Http\Controllers\Auth\LoginController;
use Illuminate\Support\Facades\Route;

// Route untuk menampilkan form login
Route::get('/login', [LoginController::class, 'showLoginForm'])->name('login');

// Route untuk memproses login
Route::post('/login', [LoginController::class, 'login']);

// Route untuk logout
Route::post('/logout', [LoginController::class, 'logout'])->name('logout');

// Route halaman setelah login (contoh)
Route::get('/dashboard', function () {
    return view('dashboard');
})->middleware('auth');

// Route home
Route::get('/', function () {
    return view('welcome');
});