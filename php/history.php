<?php 
session_start();
require_once 'DATABASE/function.php'; 

$data = $db->select('history', '*', ['user_id' => $_SESSION['user_id']]);
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>History</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
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
<?php require_once 'nav.php'; ?>     
    <div class="container mt-5"> 
        <div class="row">
            <div class="col-md-12 mt-5">
                <table id="example" class="display" style="width:100%">
                    <thead>
                        <tr>
                            <th>Inference ID</th>
                            <th>Class</th>
                            <th>Confidence</th>
                            <th>Intensity</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody>
    <?php foreach ($data as $row): 
        $info = json_decode($row['information'], true);
        $predictions = $info['predictions'];
        $first = true;
        foreach ($predictions as $prediction): 
            if ($prediction['class'] == 'strip') continue; ?>
            <tr>
                <?php if ($first): ?>
                    <td rowspan="<?php echo count($predictions) - 1; ?>"><?php echo htmlspecialchars($info['inference_id']); ?></td>
                <?php endif; ?>
                <td><?php echo htmlspecialchars($prediction['class']); ?></td>
                <td><?php echo htmlspecialchars(number_format($prediction['confidence'] * 100, 2)) . '%'; ?></td>
                <td><?php echo htmlspecialchars($prediction['intensity']); ?></td>
                <?php if ($first): ?>
                    <td rowspan="<?php echo count($predictions) - 1; ?>"><?php echo htmlspecialchars($row['created_at']); ?></td>
                <?php endif; ?>
            </tr>
        <?php 
        $first = false;
        endforeach; 
    endforeach; ?>
</tbody>

                </table>
            </div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
    <script>
        $(document).ready(function() {
            $('#example').DataTable();
        });
    </script>
</body>
</html></script>