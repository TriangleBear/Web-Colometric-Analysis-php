<?php 
session_start();
require_once 'DATABASE/function.php'; 
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $email = $_POST['email'];
    $password = $_POST['password']; // Use plain text password

    $user = $db->select('users', '*', ['email' => $email]);

    if ($user) {
        echo "<script>alert('Email already exists.');</script>";
    } else {
        $data = [
            'email' => $email,
            'password' => $password, // Store plain text password
            'type' => 0, // Assuming default user type is 0
            'created_at' => date('Y-m-d H:i:s'),
            'updated_at' => date('Y-m-d H:i:s')
        ];

        $result = $db->insert('users', $data);

        if (isset($result['status']) && $result['status'] == 'error') {
            echo "<script>alert('{$result['message']}');</script>";
        } else {
            header('Location: login.php'); // Redirect to login.php
            exit();
        }
    }
}
?>
<?php require_once 'nav.php'; ?>    

    <div class="container mt-5">
        <div class="row justify-content-center"> 
            <div class="col-md-6">
                <h1 class="text-center">Register</h1>
                <form action="register.php" method="post"> 
                    <div class="form-group"> 
                        <label for="email">Email address</label>
                        <input type="email" class="form-control" id="email" name="email" required>
                    </div>
                    <div class="form-group"> 
                        <label for="password">Password</label>
                        <input type="password" class="form-control" id="password" name="password" required>
                    </div>
                    <button type="submit" class="btn btn-primary btn-block" style="background-color: #262633; border: 2px solid white; border-radius: 20px; color: white;">Register</button>
                </form>
                <div class="text-center mt-3">
                <a href="login.php">Already have an account? Login here</a>
            </div>
            </div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>