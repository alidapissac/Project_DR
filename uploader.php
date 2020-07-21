
<?php

$target_path = "input/img.jpg"; 

if(move_uploaded_file($_FILES['uploadedfile']['tmp_name'], $target_path)) {

header("location:index2.php");
  
} else{
    echo "There was an error uploading the file, please try again!";
}
?>