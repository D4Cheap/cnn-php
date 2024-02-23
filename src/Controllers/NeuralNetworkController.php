<?php

use Psr\Container\ContainerInterface;
use Psr\Http\Message\ResponseInterface;
use Psr\Http\Message\ServerRequestInterface;

class NeuralNetworkController
{
    // constructor receives container instance
    public function __construct()
    {

    }

    public function initialize(ServerRequestInterface $request, ResponseInterface $response, array $args): ResponseInterface
    {
        // your code to access items in the container... $this->container->get('');

        $response->getBody()->write("Hello world! 2");
        return $response;
    }

    }