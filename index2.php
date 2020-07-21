<?php
error_reporting(0);
set_time_limit(0);
?>


<html>
<head>
 <meta name="viewport" content="width=device-width" content="initial-scale=1">
 <link href="css/bootstrap.min.css" rel="stylesheet">
 <link href="css/log.css" rel="stylesheet">
</head>
<body>
  <div id="p3">
  <div class="container" id="id1">
  <div class="row-justify-content-center">
  <div class="col-md-8" id='p1'>
     <h1>Diabetic Retinopathy</h1><br>
<h3>RESULT</h3>
<hr>
<center>
<img src='test/test_images/img.jpg' width='300px' height='200px' />
<br><br><br>



<?php
$python=`C:\ProgramData\Anaconda3\python.exe test.py`;
echo "<h5 > PREDICTION :- <pre style='color:#ffff;'> ".$python."</pre></h5>";
?>
 </center>
    
 <form>
     <div class="text-center"><input type="Submit" value="Back To Home" formaction="index.php"style="height: 40px; width: 250px" id="p2"></div><br>
    </form> 
   </div>
   </div>
   </div>
   </div>
   </div>
</body>
</html>