<?php
class Database
{
    // private $host = 'localhost';
    // private $db_name = 'u467106394_incidentify';
    // private $username = 'u467106394_incidentify';
    // private $password = 'Vv>qp]wGH8=n';
    private $host = 'localhost';
    private $db_name = 'colometric';
    private $username = 'root';
    private $password = '';
    public $conn;

    public function connect()
    {
        $this->conn = null;

        try {
            $this->conn = new PDO("mysql:host=" . $this->host . ";dbname=" . $this->db_name, $this->username, $this->password);
            $this->conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        } catch (PDOException $e) {
            echo "Connection error: " . $e->getMessage();
        }

        return $this->conn;
    }
}


?>