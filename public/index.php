<?php
use Psr\Http\Message\ResponseInterface as Response;
use Psr\Http\Message\ServerRequestInterface as Request;
use Slim\Factory\AppFactory;
use App\Services\FacialRecognition as FacialRecognition;

require __DIR__ . '/../vendor/autoload.php';

/*// Create Container using PHP-DI
$container = new Container();

// Set container to create App with on AppFactory
AppFactory::setContainer($container);
$app = AppFactory::create();

$container->set(NeuralNetworkController::class, function () {
    return new NeuralNetworkController();
});*/
set_time_limit(3600);

$app = AppFactory::create();

$app->get('/', [\NeuralNetworkController::class, 'initialize']);

$app->get('/train', function (Request $request, Response $response, $args) {

    $summary = FacialRecognition::train();

    $response->getBody()->write('Accuracy is '.$summary['accuracy'].', model was saved'.'\n execution time is:'.$summary['time']." seconds");

    return $response;
});

$app->get('/train/stress', function (Request $request, Response $response, $args) {
    $limit = 30;
    for ($run = 1 ; $run <= $limit; $run++)
    {
        $acc[] = FacialRecognition::train()['accuracy'];
        $time[] = FacialRecognition::train()['time'];
    }

    $sum = array_sum($acc)/count($acc);
    $sum_time = array_sum($time)/count($time);

    $response->getBody()->write("Average of model accuracy after ".$limit." trains is ".$sum.", average of time is ".$sum_time." seconds");
    return $response;
});

$app->get('/predict/image/{image}', function (Request $request, Response $response, $args) {

    $predict = FacialRecognition::predict(image: [$args['image']]);

    $response->getBody()->write('predicted is '.$predict['predict']["face"] .' with '.$predict['predict']['accuracy'].' of accuracy, time to predict was '.$predict['time']);

    return $response;
});

$app->get('/predict/plot', function (Request $request, Response $response, $args) {

    $image = FacialRecognition::predict(1, 'plot');

    ob_clean();
    header("Content-type: image/png");
    $image->show('plot.png');

    $response->getBody()->write('predicted');

    return $response;
});

$app->get('/predict/batch', function (Request $request, Response $response, $args) {

    $image = FacialRecognition::predict(8);

    ob_clean();
    header("Content-type: image/png");
    $image->show('plot.png');

    $response->getBody()->write('predicted');

    return $response;
});


$app->run();