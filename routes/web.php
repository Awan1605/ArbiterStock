<?php

use Illuminate\Support\Facades\Route;
use Illuminate\Http\Request;
use App\Models\User;
use Illuminate\Support\Facades\Hash;

/*
|--------------------------------------------------------------------------
| PUBLIC
|--------------------------------------------------------------------------
*/
Route::get('/', fn () => view('lending_page'))->name('lending_page');
Route::get('/market', fn () => view('market'))->name('market');
Route::get('/news', fn () => view('news'))->name('news');
Route::get('/about', fn () => view('about'))->name('about');

/* =======================
|  CONTACT
|  ======================= */
Route::get('/contact', fn () => view('components.contact'))->name('contact');

/* ✅ DITAMBAHKAN: TERIMA FORM CONTACT */
Route::post('/contact', function (Request $request) {

    // sementara untuk tes dulu
    // nanti bisa diganti simpan database / kirim email
    dd($request->all());

})->name('contact.store');


/*
|--------------------------------------------------------------------------
| AUTH (dummy)
|--------------------------------------------------------------------------
*/
Route::get('/login', fn () => view('auth.login'))->name('login');
Route::post('/login', fn () => null)->name('login.store');

Route::get('/register', fn () => view('auth.register'))->name('register');
Route::post('/register', fn () => null)->name('register.store');

Route::post('/logout', fn () => null)->name('logout');


/*
|--------------------------------------------------------------------------
| ADMIN
|--------------------------------------------------------------------------
*/
Route::prefix('admin')->group(function () {

    Route::get('/dashboard', fn () => view('admin.dashboard'));

    /*
    |--------------------------------------------------------------------------
    | USER MANAGEMENT
    |--------------------------------------------------------------------------
    */

    Route::get('/user', function () {
        $users = User::latest()->get();
        return view('admin.user', compact('users'));
    })->name('admin.user');

    Route::post('/user/store', function (Request $request) {

        $request->validate([
            'name'  => 'required',
            'email' => 'required|email|unique:users',
            'role'  => 'required',
        ]);

        User::create([
            'name'     => $request->name,
            'email'    => $request->email,
            'password' => Hash::make('password123'),
            'role'     => $request->role,
        ]);

        return redirect()->route('admin.user');
    })->name('admin.user.store');

    Route::put('/user/update/{id}', function (Request $request, $id) {

        $request->validate([
            'name'  => 'required',
            'email' => 'required|email',
            'role'  => 'required',
        ]);

        $user = User::findOrFail($id);

        $user->update([
            'name'  => $request->name,
            'email' => $request->email,
            'role'  => $request->role,
        ]);

        return redirect()->route('admin.user');
    })->name('admin.user.update');

    Route::post('/user/delete/{id}', function ($id) {
        User::findOrFail($id)->delete();
        return redirect()->route('admin.user');
    })->name('admin.user.delete');

    Route::get('/saham', fn () => view('admin.saham'));
    Route::get('/insight', fn () => view('admin.insight'));
});
