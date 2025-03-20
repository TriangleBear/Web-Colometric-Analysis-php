<?php 
session_start();
require_once 'DATABASE/function.php'; 

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $email = $_POST['email'];
    $password = $_POST['password'];

    $user = $db->select('users', '*', ['email' => $email]);

    if ($user) {
        if ($password == $user[0]['password']) {
            session_start();
            $_SESSION['user_id'] = $user[0]['id'];
            if ($user[0]['type'] == 0) {
                header('Location: index.php');
            } else {
                header('Location: table.php?table=history');
            }
            exit();
        } else {
            $error = "Invalid email or password.";
        }
    } else {
        $error = "Invalid email or password.";
    }
}
?>
<?php require_once 'nav.php'; ?>    
<div class="container mt-5"> 
    <div class="row justify-content-center">
        <div class="col-md-6">
            <h1 class="text-center">Login</h1>
            <?php if (isset($error)): ?>
                <div class="alert alert-danger" role="alert">
                    <?php echo $error; ?>
                </div>
            <?php endif; ?>
            <form action="login.php" method="post"> 
                <div class="form-group"> 
                    <label for="email">Email address</label>
                    <input type="email" class="form-control" id="email" name="email" required>
                </div>
                <div class="form-group"> 
                    <label for="password">Password</label>
                    <input type="password" class="form-control" id="password" name="password" required>
                </div>
                <button type="submit" class="btn btn-primary btn-block" style="background-color: #262633; border: 2px solid white; border-radius: 20px; color: white;">Login</button>
            </form>
            <div class="text-center mt-3">
                <a href="register.php">Don't have an account? Register here</a>
            </div>
        </div>
    </div>
</div>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>