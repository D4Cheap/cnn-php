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

$app->get('/testenn', function (Request $request, Response $response, $args) {
    for ($run = 1 ; $run <=5; $run++)
    {
        $acc[] = FacialRecognition::train();
    }

    $sum = array_sum($acc)/count($acc);

    $response->getBody()->write("Hello world! ".$sum);
    return $response;
});

$app->get('/teste', function (Request $request, Response $response, $args) {
    $response->getBody()->write("Hello world! 2");
    return $response;
});

$app->run();