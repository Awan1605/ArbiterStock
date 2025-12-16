@extends('admin.layout.app')

@section('content')

<!-- ================= TOPBAR ================= -->
<div class="topbar">
    <div>
        <h2>Hello, Muhammad Risky👋</h2>
        <small>Have a nice day</small>
    </div>

    <div class="profile">
        🔔
        <img src="https://i.pravatar.cc/100?img=12">
        <div>
            <strong>Muhammad Risky</strong><br>
            <small>Admin</small>
        </div>
    </div>
</div>

<!-- ================= ACTION BAR ================= -->
<div class="user-action">
    <input type="text" placeholder="Search user..." class="search">

    <div class="action-right">
        <button class="btn-primary" onclick="openAddModal()">Add Pengguna +</button>
    </div>
</div>

<!-- ================= TABLE ================= -->
<div class="table-card">
    <h3>List ArbiterStock Owner</h3>

    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Role</th>
                <th style="width:120px">Action</th>
            </tr>
        </thead>
        <tbody>

            @foreach ($users as $user)
            <tr>
                <td><strong>{{ $user->name }}</strong></td>
                <td>{{ $user->email }}</td>
                <td>
                    <span class="badge blue">{{ ucfirst($user->role) }}</span>
                </td>
                <td class="action">

                    <!-- EDIT -->
                    <button class="icon-btn"
                        onclick="openEditModal(
                            {{ $user->id }},
                            '{{ $user->name }}',
                            '{{ $user->email }}',
                            '{{ $user->role }}'
                        )">✏️</button>

                    <!-- DELETE -->
                    <form method="POST"
                          action="{{ route('admin.user.delete', $user->id) }}"
                          style="display:inline">
                        @csrf
                        <button class="icon-btn danger"
                                onclick="return confirm('Hapus user ini?')">🗑️</button>
                    </form>

                </td>
            </tr>
            @endforeach

        </tbody>
    </table>
</div>

<!-- ================= MODAL ADD USER ================= -->
<div class="modal" id="addModal">
    <div class="modal-box">
        <h3>Tambah Pengguna</h3>

        <form method="POST" action="{{ route('admin.user.store') }}">
            @csrf

            <input type="text" name="name" placeholder="Nama" required>
            <input type="email" name="email" placeholder="Email" required>

            <select name="role" required>
                <option value="">-- Pilih Role --</option>
                <option value="admin">Admin</option>
                <option value="owner">Owner</option>
                <option value="user">User</option>
            </select>

            <div class="modal-action">
                <button type="button" onclick="closeAddModal()">Batal</button>
                <button type="submit" class="btn-primary">Simpan</button>
            </div>
        </form>
    </div>
</div>

<!-- ================= MODAL EDIT USER ================= -->
<div class="modal" id="editModal">
    <div class="modal-box">
        <h3>Edit Pengguna</h3>

        <form method="POST" id="editForm">
            @csrf
            @method('PUT') {{-- 🔥 INI YANG TADI HILANG --}}

            <input type="text" name="name" id="editName" required>
            <input type="email" name="email" id="editEmail" required>

            <select name="role" id="editRole" required>
                <option value="admin">Admin</option>
                <option value="owner">Owner</option>
                <option value="user">User</option>
            </select>

            <div class="modal-action">
                <button type="button" onclick="closeEditModal()">Batal</button>
                <button type="submit" class="btn-primary">Update</button>
            </div>
        </form>
    </div>
</div>

<!-- ================= SCRIPT ================= -->
<script>
function openAddModal() {
    document.getElementById('addModal').style.display = 'flex';
}
function closeAddModal() {
    document.getElementById('addModal').style.display = 'none';
}

function openEditModal(id, name, email, role) {
    document.getElementById('editName').value = name;
    document.getElementById('editEmail').value = email;
    document.getElementById('editRole').value = role;

    document.getElementById('editForm').action =
        `/admin/user/update/${id}`;

    document.getElementById('editModal').style.display = 'flex';
}
function closeEditModal() {
    document.getElementById('editModal').style.display = 'none';
}
</script>

@endsection
