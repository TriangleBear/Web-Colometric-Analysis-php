<?php 
require_once 'Web-Colometric-Analysis-php\DATABASE\function.php'; 

$table = isset($_GET['table']) ? $_GET['table'] : '';

$data = [];
if (!empty($table)) {
    $data = $db->select($table);
}

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>URINALYZE - Dashboard</title>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #262633;
            color: white;
        }
        .sidebar {
            height: 100vh;
            width: 250px;
            position: fixed;
            top: 0;
            left: 0;
            background-color: #23232E;
            padding-top: 20px;
            transition: width 0.3s;
        }
        .sidebar a {
            padding: 15px;
            text-decoration: none;
            font-size: 18px;
            color: white;
            display: block;
            z-index: 1000; 
        }
        .sidebar a:hover {
            background-color: #575757;
        }
        .content {
            margin-left: 260px;
            padding: 20px;
        }
        .logo {
            text-align: center;
            margin-bottom: 20px;
        }
        .logo img {
            width: 100px;
            height: 100px;
        }
        .logo h1 {
            color: #F1BB65;
        }
        .table-dark, td {
            background-color: #343a40 !important;
        }
        .dataTables_info, .dataTables_length, .dataTables_filter{
            color: white !important;
        }
        .table-dark th, .table-dark td {
            color: #F1BB65;
        }
        .hamburger {
            display: none;
            position: absolute;
            top: 15px;
            left: 15px;
            cursor: pointer;
        }
        .hamburger div {
            width: 30px;
            height: 3px;
            background-color: white;
            margin: 5px 0;
        }
        @media (max-width: 768px) {
            .sidebar {
                width: 0;
                overflow: hidden;
            }
            .content {
                margin-left: 0;
            }
            .hamburger {
                display: block;
            }
        }
    </style>
</head>
<body>
    <div class="hamburger" onclick="toggleSidebar()">
        <div></div>
        <div></div>
        <div></div>
    </div>
    <div class="sidebar"> 
        <div class="logo"> 
            <img src="https://placehold.co/100" alt="Logo" style="border-radius: 50%;">
            <h1>URINALYZE</h1>
        </div>
        <a href="table.php?table=users">User List</a>
        <a href="table.php?table=history">History</a>
        <a href="logout.php">Logout</a>
    </div>
    <div class="content" > 
        <div class="container mt-5"> 
            <div class="row justify-content-center"> 
                <div class="col-md-12"> 
                    <h1 class="text-center"><?php echo ucfirst(htmlspecialchars($table)); ?> List</h1>
                    <?php if (!empty($data)): ?>
                        <table id="dataTable" class="table table-dark"> 
                            <thead>
                                <tr>
                                    <?php foreach (array_keys($data[0]) as $column): ?>
                                        <?php if ($column != 'updated_at' && $column != 'id'): ?>
                                            <th>
                                                <?php 
                                                    if ($column == 'created_at' && $table == 'history') {
                                                        echo 'Timestamp';
                                                    } else {
                                                        echo ucfirst(htmlspecialchars($column));
                                                    }
                                                ?>
                                            </th>
                                        <?php endif; ?>
                                    <?php endforeach; ?>
                                </tr>
                            </thead>
                            <tbody>
                                <?php foreach ($data as $row): ?>
                                    <tr>
                                        <?php foreach ($row as $key => $cell): ?>
                                            <?php if ($key != 'updated_at' && $key != 'id'): ?>
                                                <td>
                                                    <?php 
                                                        if ($key == 'type') {
                                                            echo $cell == 1 ? 'Admin' : 'User';
                                                        } elseif ($key == 'information') {
                                                            $info = json_decode($cell, true);
                                                            echo '<strong>Inference ID:</strong> ' . htmlspecialchars($info['inference_id']) . '<br>';
                                                            echo '<strong>Predictions:</strong><br>';
                                                            foreach ($info['predictions'] as $prediction) {
                                                                if ($prediction['class'] != 'strip') {
                                                                    echo 'Class: ' . htmlspecialchars($prediction['class']) . '<br>';
                                                                    echo 'Confidence: ' . round($prediction['confidence'] * 100) . '%<br>';
                                                                    echo 'Intensity: ' . htmlspecialchars($prediction['intensity']) . '<br>';
                                                                    echo '<hr>';
                                                                }
                                                            }
                                                        } elseif ($key == 'user_id') {
                                                            $user = $db->select('users', 'email', ['id' => $cell]);
                                                            echo htmlspecialchars($user[0]['email']);
                                                        } else {
                                                            echo htmlspecialchars($cell);
                                                        }
                                                    ?>
                                                </td>
                                            <?php endif; ?>
                                        <?php endforeach; ?>
                                    </tr>
                                <?php endforeach; ?>
                            </tbody>
                        </table>
                    <?php else: ?>
                        <p class="text-center">No data available.</p>
                    <?php endif; ?>
                </div>
            </div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#dataTable').DataTable();
        });

        function toggleSidebar() {
            var sidebar = document.querySelector('.sidebar');
            if (sidebar.style.width === '250px') {
                sidebar.style.width = '0';
            } else {
                sidebar.style.width = '250px';
            }
        }
    </script>
</body>
</html>
