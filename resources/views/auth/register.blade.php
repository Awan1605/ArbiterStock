<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Register | ArbiterStock</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --secondary: #1e293b;
      --light-gray: #f8fafc;
      --gray: #94a3b8;
      --dark-gray: #64748b;
      --success: #10b981;
      --error: #ef4444;
      --shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
      --shadow-lg: 0 20px 40px rgba(0, 0, 0, 0.1);
      --radius: 12px;
      --transition: all 0.3s ease;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: "Poppins", sans-serif;
      background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      padding: 20px;
      position: relative;
      overflow-x: hidden;
    }

    /* Background decorative elements */
    .bg-shape {
      position: absolute;
      border-radius: 50%;
      z-index: -1;
    }

    .bg-shape-1 {
      width: 300px;
      height: 300px;
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%);
      top: 5%;
      right: 5%;
    }

    .bg-shape-2 {
      width: 200px;
      height: 200px;
      background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(37, 99, 235, 0.03) 100%);
      bottom: 10%;
      left: 5%;
    }

    .brand {
      position: absolute;
      top: 30px;
      left: 40px;
      font-size: 1.8rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .brand-link {
      text-decoration: none;
      color: var(--secondary);
      display: flex;
      align-items: center;
      transition: var(--transition);
    }

    .brand-link:hover {
      color: var(--primary);
      transform: translateY(-2px);
    }

    .brand-icon {
      font-size: 1.5rem;
      color: var(--primary);
    }

    .container {
      width: 100%;
      max-width: 420px;
      background-color: white;
      border-radius: var(--radius);
      box-shadow: var(--shadow-lg);
      padding: 40px;
      animation: fadeIn 0.8s ease-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    h1 {
      font-size: 2.2rem;
      margin-bottom: 8px;
      color: var(--secondary);
      text-align: center;
    }

    .subtitle {
      color: var(--dark-gray);
      text-align: center;
      margin-bottom: 30px;
      font-weight: 400;
      font-size: 0.95rem;
    }

    .form-group {
      margin-bottom: 22px;
      position: relative;
    }

    label {
      display: block;
      text-align: left;
      font-size: 0.9rem;
      margin-bottom: 8px;
      color: var(--secondary);
      font-weight: 500;
    }

    .input-container {
      position: relative;
    }

    input {
      width: 100%;
      padding: 14px 14px 14px 45px;
      border: 1.5px solid #e2e8f0;
      border-radius: 8px;
      font-size: 0.95rem;
      transition: var(--transition);
      background-color: var(--light-gray);
    }

    input:focus {
      outline: none;
      border-color: var(--primary);
      background-color: white;
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    .input-icon {
      position: absolute;
      left: 15px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--gray);
      font-size: 1rem;
    }

    .password-toggle {
      position: absolute;
      right: 15px;
      top: 50%;
      transform: translateY(-50%);
      background: none;
      border: none;
      color: var(--gray);
      cursor: pointer;
      font-size: 1rem;
    }

    .password-toggle:hover {
      color: var(--dark-gray);
    }

    .validation-message {
      font-size: 0.8rem;
      margin-top: 6px;
      display: flex;
      align-items: center;
      gap: 5px;
      opacity: 0;
      transition: var(--transition);
    }

    .validation-message.show {
      opacity: 1;
    }

    .validation-message.valid {
      color: var(--success);
    }

    .validation-message.invalid {
      color: var(--error);
    }

    .small-text {
      font-size: 0.85rem;
      color: var(--dark-gray);
      margin-bottom: 25px;
      text-align: center;
    }

    .small-text a {
      color: var(--primary);
      text-decoration: none;
      font-weight: 500;
      transition: var(--transition);
    }

    .small-text a:hover {
      text-decoration: underline;
    }

    .btn {
      width: 100%;
      padding: 15px;
      background: linear-gradient(to right, var(--primary), var(--primary-dark));
      color: white;
      font-weight: 600;
      font-size: 1rem;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: var(--transition);
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 10px;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    .btn:hover {
      background: linear-gradient(to right, var(--primary-dark), var(--primary));
      transform: translateY(-2px);
      box-shadow: 0 6px 15px rgba(37, 99, 235, 0.3);
    }

    .btn:active {
      transform: translateY(0);
    }

    .divider {
      display: flex;
      align-items: center;
      margin: 25px 0;
      color: var(--dark-gray);
      font-size: 0.85rem;
    }

    .divider::before,
    .divider::after {
      content: "";
      flex: 1;
      border-bottom: 1px solid #e2e8f0;
    }

    .divider::before {
      margin-right: 15px;
    }

    .divider::after {
      margin-left: 15px;
    }

    .social-login {
      display: flex;
      justify-content: center;
      gap: 15px;
      margin-bottom: 25px;
    }

    .social-btn {
      width: 50px;
      height: 50px;
      border-radius: 50%;
      border: 1.5px solid #e2e8f0;
      background-color: white;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      transition: var(--transition);
    }

    .social-btn:hover {
      background-color: var(--light-gray);
      transform: translateY(-2px);
      border-color: var(--gray);
    }

    .social-btn i {
      font-size: 1.2rem;
    }

    .google i {
      color: #ea4335;
    }

    .github i {
      color: #333;
    }

    .twitter i {
      color: #1da1f2;
    }

    .terms {
      font-size: 0.75rem;
      color: var(--dark-gray);
      text-align: center;
      margin-top: 20px;
    }

    .terms a {
      color: var(--primary);
      text-decoration: none;
    }

    @media (max-width: 768px) {
      .container {
        padding: 30px 25px;
        max-width: 380px;
      }
      
      .brand {
        left: 20px;
        top: 20px;
        font-size: 1.5rem;
      }
      
      h1 {
        font-size: 1.9rem;
      }
    }

    @media (max-width: 480px) {
      .container {
        padding: 25px 20px;
      }
      
      .brand {
        position: relative;
        top: 0;
        left: 0;
        justify-content: center;
        margin-bottom: 20px;
      }
      
      h1 {
        font-size: 1.7rem;
      }
      
      .social-login {
        gap: 10px;
      }
      
      .social-btn {
        width: 45px;
        height: 45px;
      }
    }
  </style>
</head>
<body>
  <!-- Background decorative shapes -->
  <div class="bg-shape bg-shape-1"></div>
  <div class="bg-shape bg-shape-2"></div>

  <a href="/" class="brand-link">
    <div class="brand">
      <i class="fas fa-chart-line brand-icon"></i>
      ArbiterStock
    </div>
  </a>

  <div class="container">
    <h1>Create Account</h1>
    <p class="subtitle">Join ArbiterStock and start your trading journey</p>

    <form id="registerForm" action="{{ route('register.store') }}" method="POST">
      @csrf
      
      <div class="form-group">
        <label for="username">Username</label>
        <div class="input-container">
          <i class="fas fa-user input-icon"></i>
          <input type="text" id="username" name="username" placeholder="Enter your username" required>
        </div>
        <div id="username-validation" class="validation-message">
          <i class="fas fa-check-circle"></i>
          <span>Username is available</span>
        </div>
      </div>

      <div class="form-group">
        <label for="email">Email Address</label>
        <div class="input-container">
          <i class="fas fa-envelope input-icon"></i>
          <input type="email" id="email" name="email" placeholder="Enter your email" required>
        </div>
        <div id="email-validation" class="validation-message">
          <i class="fas fa-check-circle"></i>
          <span>Valid email format</span>
        </div>
      </div>

      <div class="form-group">
        <label for="password">Password</label>
        <div class="input-container">
          <i class="fas fa-lock input-icon"></i>
          <input type="password" id="password" name="password" placeholder="Create a strong password" required>
          <button type="button" class="password-toggle" id="togglePassword">
            <i class="far fa-eye"></i>
          </button>
        </div>
        <div id="password-validation" class="validation-message">
          <i class="fas fa-info-circle"></i>
          <span>Password must be at least 8 characters</span>
        </div>
      </div>

      <div class="small-text">
        Already have an account? <a href="{{ route('login') }}">Sign in here</a>
      </div>

      <button type="submit" class="btn">
        <i class="fas fa-user-plus"></i>
        Create Account
      </button>

      <div class="divider">Or continue with</div>

      <div class="social-login">
        <div class="social-btn google">
          <i class="fab fa-google"></i>
        </div>
        <div class="social-btn github">
          <i class="fab fa-github"></i>
        </div>
        <div class="social-btn twitter">
          <i class="fab fa-twitter"></i>
        </div>
      </div>

      <p class="terms">
        By registering, you agree to our <a href="#">Terms of Service</a> and <a href="#">Privacy Policy</a>.
      </p>
    </form>
  </div>

  <script>
    // Password visibility toggle
    const togglePassword = document.getElementById('togglePassword');
    const passwordInput = document.getElementById('password');
    const eyeIcon = togglePassword.querySelector('i');
    
    togglePassword.addEventListener('click', function() {
      const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
      passwordInput.setAttribute('type', type);
      eyeIcon.classList.toggle('fa-eye');
      eyeIcon.classList.toggle('fa-eye-slash');
    });

    // Form validation
    const registerForm = document.getElementById('registerForm');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    
    const usernameValidation = document.getElementById('username-validation');
    const emailValidation = document.getElementById('email-validation');
    const passwordValidation = document.getElementById('password-validation');
    
    // Username validation
    usernameInput.addEventListener('input', function() {
      const username = this.value;
      const usernameRegex = /^[a-zA-Z0-9_]{3,20}$/;
      
      if (username.length === 0) {
        hideValidation(usernameValidation);
        return;
      }
      
      if (username.length < 3) {
        showValidation(usernameValidation, 'Username must be at least 3 characters', false);
      } else if (!usernameRegex.test(username)) {
        showValidation(usernameValidation, 'Only letters, numbers and underscores allowed', false);
      } else {
        setTimeout(() => {
          showValidation(usernameValidation, 'Username is available', true);
        }, 300);
      }
    });
    
    // Email validation
    emailInput.addEventListener('input', function() {
      const email = this.value;
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      
      if (email.length === 0) {
        hideValidation(emailValidation);
        return;
      }
      
      if (emailRegex.test(email)) {
        showValidation(emailValidation, 'Valid email format', true);
      } else {
        showValidation(emailValidation, 'Please enter a valid email address', false);
      }
    });
    
    // Password validation
    passwordInput.addEventListener('input', function() {
      const password = this.value;
      
      if (password.length === 0) {
        hideValidation(passwordValidation);
        return;
      }
      
      let message = '';
      let isValid = true;
      
      if (password.length < 8) {
        message = 'Password must be at least 8 characters';
        isValid = false;
      } else if (!/\d/.test(password)) {
        message = 'Password should contain at least one number';
        isValid = false;
      } else if (!/[A-Z]/.test(password)) {
        message = 'Password should contain at least one uppercase letter';
        isValid = false;
      } else {
        message = 'Strong password';
        isValid = true;
      }
      
      showValidation(passwordValidation, message, isValid);
    });
    
    function showValidation(element, message, isValid) {
      const icon = element.querySelector('i');
      const text = element.querySelector('span');
      
      text.textContent = message;
      element.classList.remove('valid', 'invalid');
      element.classList.add(isValid ? 'valid' : 'invalid');
      element.classList.add('show');
      
      // Update icon
      if (isValid) {
        icon.className = 'fas fa-check-circle';
      } else {
        icon.className = 'fas fa-exclamation-circle';
      }
    }
    
    function hideValidation(element) {
      element.classList.remove('show', 'valid', 'invalid');
    }
    
    // Social buttons interaction
    const socialButtons = document.querySelectorAll('.social-btn');
    socialButtons.forEach(button => {
      button.addEventListener('click', function() {
        const platform = this.classList.contains('google') ? 'Google' : 
                        this.classList.contains('github') ? 'GitHub' : 'Twitter';
        alert(`You clicked on ${platform} login. This is a demo feature.`);
        
        // Add animation effect
        this.style.transform = 'scale(0.95)';
        setTimeout(() => {
          this.style.transform = '';
        }, 150);
      });
    });
    
    // Form submission
    registerForm.addEventListener('submit', function(e) {
      e.preventDefault();
      
      const submitBtn = this.querySelector('.btn');
      const originalText = submitBtn.innerHTML;
      
      submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating Account...';
      submitBtn.disabled = true;
      
    
      setTimeout(() => {
        alert('Registration successful! In a real app, this would redirect to the dashboard.');
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        
        // In a real app, you would redirect or show a success message
        // window.location.href = '/dashboard';
      }, 1500);
    });
    
    // Add focus effect to inputs
    const inputs = document.querySelectorAll('input');
    inputs.forEach(input => {
      input.addEventListener('focus', function() {
        this.parentElement.style.transform = 'translateY(-2px)';
      });
      
      input.addEventListener('blur', function() {
        this.parentElement.style.transform = 'translateY(0)';
      });
    });
  </script>
</body>
</html>