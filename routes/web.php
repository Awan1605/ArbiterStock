<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Auth\AuthController;

/*
|--------------------------------------------------------------------------
| Routes untuk Otentikasi (Auth)
|--------------------------------------------------------------------------
*/

// Tampilan Form Login
Route::get('/login', [AuthController::class, 'showLoginForm'])->name('login');

// Proses Login
Route::post('/login', [AuthController::class, 'login']);

// Tampilan Form Register
Route::get('/register', [AuthController::class, 'showRegistrationForm'])->name('register'); // ⬅️ TELAH DIPERBAIKI (menjadi showRegistrationForm)

// Proses Register
Route::post('/register', [AuthController::class, 'register']);

// Proses Logout
Route::post('/logout', [AuthController::class, 'logout'])->name('logout');

/*
|--------------------------------------------------------------------------
| Route Default (Jika ada)
|--------------------------------------------------------------------------
*/
// Route::get('/', function () {
//     return view('welcome');
// });