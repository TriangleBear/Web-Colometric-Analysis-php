<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>URINALYZE - Login</title>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #262633;
            color: white;
        }
        .navbar {
            background-color: #23232E;
        }
        .navbar-brand {
            color: white;
        }
        .nav-link {
            color: white;
        }
        .navbar-toggler-icon {
            background-image: url("data:image/svg+xml;charset=utf8,%3Csvg viewBox='0 0 30 30' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath stroke='rgba%28255, 255, 255, 1%29' stroke-width='2' stroke-linecap='round' stroke-miterlimit='10' d='M4 7h22M4 15h22M4 23h22'/%3E%3C/svg%3E");
        }
    </style>
</head>
<body>
<nav class="navbar navbar-expand-lg">
        <a style="color: #F1BB65" class="navbar-brand" href="index.php">URINALYZE</a>
        <button  class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ml-auto">
                <li class="nav-item">
                    <a class="nav-link" href="index.php">Home</a>
                                    </li>
                                    <li class="nav-item">
                                        <?php  if(isset($_SESSION['user_id'])): ?>
                                            <a class="nav-link" href="logout.php">Logout</a>
                                        <?php else: ?>
                                            <a class="nav-link" href="login.php">Login</a>
                                        <?php endif; ?>
                                    </li>   
                                    <li class="nav-item"> 
                                        <a class="nav-link" href="<?php echo isset($_SESSION['user_id']) ? 'upload.php' : 'login.php'; ?>">Strip Analysis</a>
                                    </li>
                                    <li class="nav-item"> 
                                        <a class="nav-link" href="<?php echo isset($_SESSION['user_id']) ? 'history.php' : 'login.php'; ?>">History</a>
                                    </li>
            </ul>
        </div>
    </nav>  