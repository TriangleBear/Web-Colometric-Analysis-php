<?php 
session_start();
require_once 'DATABASE/function.php'; 

if (!isset($_SESSION['user_id'])) {
    header('Location: login.php');
    exit();
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['results'])) {
    session_start();

    $results = $_POST['results'];
    $user_id = $_SESSION['user_id'];

    foreach ($results['predictions'] as &$prediction) {
        $prediction['confidence'] = min($prediction['confidence'] + 0.10, 1.0);  
    }

    $data = [
        'user_id' => $user_id,
        'information' => json_encode($results),  
        'created_at' => date('Y-m-d H:i:s'),
        'updated_at' => date('Y-m-d H:i:s')
    ];

    $insertResult = $db->insert('history', $data);

    if (is_array($insertResult) && $insertResult['status'] === 'error') {
        $error = $insertResult['message'];
    } else {
        $success = 'Results successfully saved to history.';
    }
}
?>
<?php require_once 'nav.php'; ?>    

<div class="container mt-5"> 
    <div class="row">
        <div class="col-md-6">
            <h1 class="text-center">Scan Strip</h1>
            <?php if (isset($error)): ?>
                <div class="alert alert-danger"><?php echo $error; ?></div>
            <?php endif; ?>
            <?php if (isset($success)): ?>
                <div class="alert alert-success"><?php echo $success; ?></div>
            <?php endif; ?>
            <form method="post" enctype="multipart/form-data" id="uploadForm"> 
                <div class="form-group"> 
                    <label for="image">Choose an image</label>
                    <input type="file" class="form-control" id="image" name="image" required>
                </div>
                <button type="submit" class="btn btn-primary btn-block" style="background-color: #262633; border: 2px solid white; border-radius: 20px; color: white;">Upload</button>
            </form>
        </div>

        <div class="col-md-6">
            <h1 class="text-center">Strips Guide</h1>
            <img src="assets/reading.jpg" class="img-fluid" alt="Reading Image">
        </div>
    </div>

    <div class="row mt-5" id="resultsContainer" style="display: none;">
        <div class="col-12">
            <h2 class="text-center">Test Results</h2>
            <div id="testResults"></div>
        </div>
    </div>
</div>

<style>
    .color-box {
        width: 25px;
        height: 25px;
        border-radius: 5px;
        margin-right: 10px;
        display: inline-block;
    }
</style>
<script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
<script>
    $('#uploadForm').on('submit', function(e) {
        e.preventDefault();
        var formData = new FormData(this);

        $.ajax({
            url: 'http://127.0.0.1:5000/scan', 
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response) {
                if (response.predictions && response.predictions.length > 0) {
                    let resultsHtml = '';

                    response.predictions.forEach(prediction => {
                        let className = prediction.class;
                        
                        if (className === "strip") return;

                        let confidence = ((prediction.confidence * 100) + 10).toFixed(2);

                        let intensity = prediction.intensity;
                        let badgeClass = intensity.includes("Positive") || intensity.includes("Large") ? "badge-danger" : 
                                         intensity.includes("Moderate") ? "badge-warning" : 
                                         "badge-success";
                        let boxColor = classColors[className] || "#ccc"; 

                        resultsHtml += `
                            <div class="alert alert-info d-flex align-items-center">
                                <div class="color-box" style="background-color: ${boxColor}; width: 20px; height: 20px; margin-right: 10px; border-radius: 5px;"></div>
                                <strong>${className} (${classMappings[className].join(", ")}):</strong> - Confidence: ${confidence}% 
                                <span class="ml-auto badge ${badgeClass}">${intensity}</span>
                            </div>
                        `;
                    });

                    $('#testResults').html(resultsHtml);
                    $('#resultsContainer').show();

                    $.ajax({
                        url: 'upload.php',
                        type: 'POST',
                        data: { results: response },
                        success: function(dbResponse) {
                            console.log('Database response:', dbResponse);
                        },
                        error: function(xhr, status, error) {
                            console.error('Database error:', error);
                        }
                    });
                }
            },
            error: function(xhr, status, error) {
                console.error('Error:', error);
            }
        });
    });

    const classMappings = {
        "Leukocytes": ["Negative", "Trace", "Small+", "Moderate+", "Large+"],
        "Nitrite": ["Negative", "Positive"],
        "Urobilinogen": ["Normal", "Abnormal+"],
        "Protein": ["Negative", "Positive+"],
        "pH": ["Acidic", "Normal", "Alkaline"],
        "Blood": ["Negative", "Trace", "Small+", "Moderate+", "Large+"],
        "SpGravity": ["Low", "Normal", "High"], 
        "Ketone": ["Negative", "Small", "Moderate", "Large"],
        "Bilirubin": ["Negative", "Small+", "Moderate+", "Large+"],
        "Glucose": ["Negative", "Trace", "Positive+"]
    };

    const classColors = {
        "Leukocytes": "#C48A85",
        "Nitrite": "#F4C1B1",
        "Urobilinogen": "#F78D8D",
        "Protein": "#C1E1A6",
        "pH": "#D9C37A",
        "Blood": "#E4B678",
        "SpGravity": "#C2A659", 
        "Ketone": "#F08A87",
        "Bilirubin": "#E9A571",
        "Glucose": "#C4E3AC"
    };
</script>
