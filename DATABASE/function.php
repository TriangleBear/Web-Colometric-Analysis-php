<?php
include_once 'db.php';
require 'vendor/autoload.php';
error_reporting(E_ALL);
ini_set('display_errors', 0);
ini_set('log_errors', 1);

use PHPMailer\PHPMailer\PHPMailer;
use PHPMailer\PHPMailer\Exception;

$db = new DBFunctions();

class DBFunctions
{
    private $conn;
    private $api_token = '870|h05YLghELQ8xSwBYKosPFx3w6svYs4EckHpQvsf9 ';
    private $debug = 1;
    public function __construct()
    {
        $database = new Database();
        $this->conn = $database->connect();
    }

    private function checkSqlError($stmt)
    {
        if ($this->debug == 1) {
            $errorInfo = $stmt->errorInfo();
            if ($errorInfo[0] != '00000') {
                return ['status' => 'error', 'message' => 'SQL Error: ' . $errorInfo[2]];
            }
        }
        return null;
    }

    public function insert($table, $data)
    {
        try {
            $columns = implode(", ", array_keys($data));
            $placeholders = ":" . implode(", :", array_keys($data));

            $sql = "INSERT INTO $table ($columns) VALUES ($placeholders)";
            $stmt = $this->conn->prepare($sql);

            foreach ($data as $key => &$val) {
                $stmt->bindParam(':' . $key, $val);
            }

            $stmt->execute();
            
            $error = $this->checkSqlError($stmt);
            if ($error) {
                return $error;
            }
        } catch (PDOException $e) {
            return ['status' => 'error', 'message' => 'Error: ' . $e->getMessage()];
        }
    }

    public function update($table, $data, $conditions = [])
{
    $set_clause = implode(", ", array_map(function ($key) {
        return "$key = :$key";
    }, array_keys($data)));

    $condition_clause = implode(" AND ", array_map(function ($key, $value) {
        // Handle raw SQL expressions
        if (is_string($value) && strpos($value, 'RAW:') === 0) {
            return substr($value, 4); // Remove the 'RAW:' prefix
        } elseif (is_array($value) && count($value) === 2) {
            return "$key {$value[0]} :$key";
        }
        return "$key = :$key";
    }, array_keys($conditions), $conditions));

    $sql = "UPDATE $table SET $set_clause WHERE $condition_clause";
    $stmt = $this->conn->prepare($sql);

    foreach ($data as $key => &$val) {
        $stmt->bindParam(':' . $key, $val);
    }

    foreach ($conditions as $key => &$val) {
        if (is_array($val)) {
            $stmt->bindParam(':' . $key, $val[1]);
        } elseif (strpos($val, 'RAW:') === false) {
            $stmt->bindParam(':' . $key, $val);
        }
    }

    $stmt->execute();
    $error = $this->checkSqlError($stmt);
    if ($error) {
        return $error;
    }
    return true;
}


    public function delete($table, $conditions)
    {
        $condition_clause = implode(" AND ", array_map(function ($key) {
            return "$key = :$key";
        }, array_keys($conditions)));

        $sql = "DELETE FROM $table WHERE $condition_clause";
        $stmt = $this->conn->prepare($sql);

        foreach ($conditions as $key => &$val) {
            $stmt->bindParam(':' . $key, $val);
        }

        $stmt->execute();
        $error = $this->checkSqlError($stmt);
        if ($error) {
            return $error;
        }
        return true;
    }

    public function select($table, $columns = "*", $conditions = [], $options = "")
{
    $sql = "SELECT $columns FROM $table";

    if (!empty($conditions)) {
        $condition_clause = implode(" AND ", array_map(function ($key) {
            return "$key = :$key";
        }, array_keys($conditions)));

        $sql .= " WHERE $condition_clause";
    }

    if (!empty($options)) {
        $sql .= " $options";
    }

    $stmt = $this->conn->prepare($sql);

    if (!empty($conditions)) {
        foreach ($conditions as $key => &$val) {
            $stmt->bindParam(':' . $key, $val);
        }
    }

    $stmt->execute();
    $error = $this->checkSqlError($stmt);
    if ($error) {
        return $error;
    }
    return $stmt->fetchAll(PDO::FETCH_ASSOC);
}


    public function select2($table, $columns = "*", $conditions = [], $customCondition = "")
    {

        $sql = "SELECT $columns FROM $table";

        if (!empty($conditions)) {
            $condition_clause = implode(" AND ", array_map(function ($key) {
                return "$key = :$key";
            }, array_keys($conditions)));

            $sql .= " WHERE $condition_clause";
        }

        if ($customCondition) {

            $sql .= empty($conditions) ? " WHERE $customCondition" : " AND $customCondition";
        }

        $stmt = $this->conn->prepare($sql);

        if (!empty($conditions)) {
            foreach ($conditions as $key => &$val) {
                $stmt->bindParam(':' . $key, $val);
            }
        }

        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    public function getAllByTableName($table_name)
    {

        if (!preg_match('/^[a-zA-Z0-9_]+$/', $table_name)) {
            throw new InvalidArgumentException('Invalid table name');
        }

        $query = "SELECT * FROM `$table_name`";
        $stmt = $this->conn->prepare($query);

        if (!$stmt) {
            throw new RuntimeException('Failed to prepare statement: ' . $this->conn->errorInfo()[2]);
        }

        $success = $stmt->execute();

        if (!$success) {
            throw new RuntimeException('Failed to execute query: ' . $stmt->errorInfo()[2]);
        }

        $results = $stmt->fetchAll(PDO::FETCH_ASSOC);

        return $results;
    }

    public function count($table_name, $condition = null)
    {

        if (!preg_match('/^[a-zA-Z0-9_]+$/', $table_name)) {
            throw new InvalidArgumentException('Invalid table name');
        }

        $sql = "SELECT COUNT(*) AS count FROM `$table_name`";

        if ($condition) {
            $sql .= " WHERE $condition";
        }

        $stmt = $this->conn->prepare($sql);

        if (!$stmt) {
            throw new RuntimeException('Failed to prepare statement: ' . implode(' ', $this->conn->errorInfo()));
        }

        if (!$stmt->execute()) {
            throw new RuntimeException('Failed to execute query: ' . implode(' ', $stmt->errorInfo()));
        }

        $result = $stmt->fetch(PDO::FETCH_ASSOC);

        if (!$result) {
            throw new RuntimeException('Failed to fetch result: ' . implode(' ', $stmt->errorInfo()));
        }

        return $result['count'] ?? 0;
    }

    public function sum($table_name, $column_name, $condition = null)
    {

        if (!preg_match('/^[a-zA-Z0-9_]+$/', $table_name) || !preg_match('/^[a-zA-Z0-9_]+$/', $column_name)) {
            throw new InvalidArgumentException('Invalid table name or column name');
        }

        $sql = "SELECT SUM($column_name) AS total FROM `$table_name`";

        if ($condition) {
            $sql .= " WHERE $condition";
        }

        $stmt = $this->conn->prepare($sql);

        if (!$stmt) {
            throw new RuntimeException('Failed to prepare statement: ' . implode(' ', $this->conn->errorInfo()));
        }

        if (!$stmt->execute()) {
            throw new RuntimeException('Failed to execute query: ' . implode(' ', $stmt->errorInfo()));
        }

        $result = $stmt->fetch(PDO::FETCH_ASSOC);

        if (!$result) {
            throw new RuntimeException('Failed to fetch result: ' . implode(' ', $stmt->errorInfo()));
        }

        return $result['total'] ?? 0;
    }

    public function getSMSBalance()
    {
        $url = 'https://app.philsms.com/api/v3/balance';
        $headers = [
            "Authorization: Bearer {$this->api_token}",
            "Content-Type: application/json",
            "Accept: application/json"
        ];

        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
        $response = curl_exec($ch);
        curl_close($ch);

        $result = json_decode($response, true);

        if ($result === null && json_last_error() !== JSON_ERROR_NONE) {
            return 'Error decoding JSON: ' . json_last_error_msg();
        }

        if (isset($result['status']) && $result['status'] === 'success') {
            $balance = $result['data']['remaining_balance'] ?? 'N/A';
            $expiration = $result['data']['expired_on'] ?? 'N/A';

            return "{$balance}";
        } else {
            return 'Error retrieving balance: ' . ($result['message'] ?? 'Unknown error');
        }
    }

    public function sendemail($to, $message, $subject)
    {
        $mail = new PHPMailer(true);
        try {
            $mail->isSMTP();
            $mail->Host = 'smtp.gmail.com';
            $mail->SMTPAuth = true;
            $mail->Username = 'sharmaineeunice07@gmail.com';
            $mail->Password = 'fzyo optp wtjg yryl';
            $mail->SMTPSecure = 'tls';
            $mail->Port = 587;

            $mail->SMTPOptions = [
                'ssl' => [
                    'verify_peer' => false,
                    'verify_peer_name' => false,
                    'allow_self_signed' => true,
                ],
            ];

            $mail->setFrom('pnpsectortacloban@gmail.com', 'Shoppeep');
            $mail->addAddress($to);

            $mail->isHTML(true);
            $mail->Subject = $subject;
            $mail->Body = $message;

            $mail->send();
            return ['status' => 'success', 'message' => 'Email sent successfully.'];
        } catch (Exception $e) {
            return ['status' => 'error', 'message' => 'Mail Error: ' . $e->getMessage()];
        }
    }

    public function sendSMS($mobileNumber, $message)
    {
        $url = 'https://app.philsms.com/api/v3/sms/send';

        $data = [
            'recipient' => $mobileNumber,
            'sender_id' => 'PhilSMS',
            'type' => 'plain',
            'message' => $message
        ];

        $headers = [
            "Authorization: Bearer {$this->api_token}",
            "Content-Type: application/json",
            "Accept: application/json"
        ];

        $ch = curl_init();
        curl_setopt($ch, CURLOPT_URL, $url);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
        $response = curl_exec($ch);
        $error = curl_error($ch);
        curl_close($ch);

        if ($error) {
            return ['status' => 'error', 'message' => $error];
        }

        return json_decode($response, true);
    }

}