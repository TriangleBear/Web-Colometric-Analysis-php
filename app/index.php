<?php 
session_start();
?>
 
<?php require_once 'nav.php'; ?>     
    <div class="container mt-5">
        <div class="row">
            <div class="col-md-6 d-flex flex-column justify-content-center"> 
                <h1 class="text-center">Hi! Analyze Urine Test Strip</h1>
                <?php if (isset($_SESSION['user_id'])): ?>
                    <button class="btn mt-3" style="background-color: #262633; border: 2px solid white; border-radius: 20px; color: white;" onclick="window.location.href='upload.php'">Get Started</button>
                <?php else: ?>
                    <button class="btn mt-3" style="background-color: #262633; border: 2px solid white; border-radius: 20px; color: white;" onclick="window.location.href='login.php'">Get Started</button>
                <?php endif; ?>
            </div>
            <div class="col-md-6 mt-5 d-flex justify-content-center"> 
                <img src="assets/nurse.png" class="img-fluid" alt="Nurse" style="max-width: 40%;">
            </div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html> 