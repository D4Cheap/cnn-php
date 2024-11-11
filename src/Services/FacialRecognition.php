<?php

namespace App\Services;

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Model\Sequential;


class FacialRecognition{
    protected $filePath = '../src/Services/classnames.txt';

    public static function train()
    {
        $start_time = microtime(true);
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $samples = $labels = [];
        
        $dataset = $nn->data()->ImageClassifiedDataset(
            '../src/Services/training',
            pattern: '@.*\\.jpg@',
            batch_size: 32,
            height: 64,
            width: 64,
            shuffle: true);

        $dataset_testing = $nn->data()->ImageClassifiedDataset(
            '../src/Services/testing',
            pattern: '@.*\\.jpg@',
            batch_size: 32,
            height: 64,
            width: 64,
            shuffle: true);


        [$training, $training_labels] = $dataset->loadData();
        [$testing, $testing_labels] = $dataset_testing->loadData();

        echo ('images: ' . implode(',', $training->shape()) . "\n");
        echo ('labels: ' . implode(',', $training_labels->shape()) . "\n");
        echo ('images: ' . implode(',', $testing->shape()) . "\n");
        echo ('labels: ' . implode(',', $testing_labels->shape()) . "\n");

        $classnames = $dataset->classnames();



        $pltCfg = [
            'title.position' => 'down', 'title.margin' => 0,
        ];

        $plt = new Plot($pltCfg, $mo);
        $images = $training[[0, 24]];
        $labels = $training_labels[[0, 24]];

        //Plot is messed up, have to randomize images or rearrange the array
//        [$fig, $axes] = $plt->subplots(5, 5);
//        foreach ($images as $i => $image) {
//            $axes[$i]->imshow($image,
//                null, null, null, $origin = 'upper');
//            $label = $labels[$i];
//            $axes[$i]->setTitle($classnames[$label] . "($label)");
//            $axes[$i]->setFrame(false);
//        }
        //ob_clean();
        //header("Content-type: image/png");
        //$plt->show();

        $f_train_img = $mo->scale(1.0 / 255.0, $mo->la()->astype($training, NDArray::float32));
        $f_val_img = $mo->scale(1.0 / 255.0, $mo->la()->astype($testing, NDArray::float32));
        file_put_contents('../src/Services/val_img.txt', serialize($f_val_img));
        $i_train_label = $mo->la()->astype($training_labels, NDArray::int32);

        $i_train_label = $mo->la()->onehot($i_train_label, 16);

        $i_val_label = $mo->la()->astype($testing_labels, NDArray::int32);
        file_put_contents('../src/Services/val_label.txt', serialize($i_val_label));
        $i_val_label = $mo->la()->onehot($i_val_label, 16);

        file_put_contents('../src/Services/classnames.txt', serialize($classnames));

        $inputShape = $training->shape();
        array_shift($inputShape);

        //Initialize the neural network
        $model = $nn->models()->Sequential([
                $nn->layers()->Conv2D(
                    filters: 32,
                    kernel_size: [5, 5],
                    strides: [1, 1],
                    input_shape: [64, 64, 3],
                    activation: 'relu'),
                $nn->layers()->MaxPooling2D(
                    pool_size: [2, 2],),
                $nn->layers()->Conv2D(
                    filters: 64,
                    kernel_size: [5, 5],
                    strides: [1, 1],
                    activation: 'relu'),
                $nn->layers()->MaxPooling2D(
                    pool_size: [2, 2],),
                $nn->layers()->Flatten(),
                $nn->layers()->Dense(
                    units: 64,
                    activation: 'relu'
                ),
                $nn->layers()->Dense(
                    units: 16,
                    activation: 'softmax'
                ),
            ]
        );

        $model->compile(
            optimizer: 'adam',
        );

        $model->summary();


        $model->compile(
            loss: $nn->losses()->CategoricalCrossEntropy(),
            optimizer: 'adam',
        );
        $history = $model->fit($f_train_img, $i_train_label,
            epochs: 10,
            validation_data: [$f_val_img, $i_val_label]);

        foreach ($history['accuracy'] as $acc){

                echo($acc.'.......................');

        }

//
//        $plt->setConfig([]);
//        $plt->plot($mo->array($history['accuracy']), null, null, 'accuracy');
//        $plt->plot($mo->array($history['val_accuracy']), null, null, 'val_accuracy');
//        $plt->plot($mo->array($history['loss']), null, null, 'loss');
//        $plt->plot($mo->array($history['val_loss']), null, null, 'val_loss');
//        $plt->legend();
//        $plt->title('face_recognition');
        //ob_clean();
        //header("Content-type: image/png");
        //$plt->show();

        $acc = end($history['accuracy']);

        $images_j = $f_val_img[[0, 7]];

        $i_val_label = $mo->la()->astype($testing_labels, NDArray::int32);
        $labels_j = $i_val_label[[0, 7]];

        $model->save('../cnn.model');

        $end_time = microtime(true);

        $exec_time = $end_time-$start_time;

        return ['accuracy' => $acc, 'time' => $exec_time];
    }

    public static function predict(int $size = 1, $mode = 'text', ?array $image = null)
    {
        $start_time = microtime(true);

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $samples = $labels = [];

        $dataset = $nn->data()->ImageClassifiedDataset(
            '../src/Services/training',
            pattern: '@.*\\.jpg@',
            restricted_by_class: $image,
            batch_size: 32,
            height: 64,
            width: 64,
            shuffle: true);

        $dataset_testing = $nn->data()->ImageClassifiedDataset(
            '../src/Services/testing',
            pattern: '@.*\\.jpg@',
            restricted_by_class: $image,
            batch_size: 32,
            height: 64,
            width: 64,
            shuffle: true);


        [$training, $training_labels] = $dataset->loadData();
        [$testing, $testing_labels] = $dataset_testing->loadData();


        $classnames = $dataset_testing->classnames();

        $pltCfg = [
            'title.position' => 'down', 'title.margin' => 0,
        ];

        $plt = new Plot($pltCfg, $mo);



        $f_train_img = $mo->scale(1.0 / 255.0, $mo->la()->astype($training, NDArray::float32));
        $f_val_img = $mo->scale(1.0 / 255.0, $mo->la()->astype($testing, NDArray::float32));
        $i_train_label = $mo->la()->astype($training_labels, NDArray::int32);
        $i_train_label = $mo->la()->onehot($i_train_label, 16);

        $i_val_label = $mo->la()->astype($testing_labels, NDArray::int32);



        $inputShape = $training->shape();
        array_shift($inputShape);

        $model = $nn->models()->loadModel('../cnn.model');

        $select = [];

        //grab 8 random pictures for the predict data

        if($image){
            $select[] = 0;
        } else {
            for ($i = 1; $i <= $size; $i++) {
                $select[] = random_int(0, 63);

            }
        }

        $indexPredict = $mo->array(
            $select,NDArray::int32);

        $images_j = $mo->select($f_val_img,$indexPredict);

        $labels_j = $mo->select($i_val_label, $indexPredict);


        $predicts = $model->predict($images_j);


        $og_classnames = unserialize(file_get_contents('../src/Services/classnames.txt'));

        if ($mode == 'plot') {
            $plt->setConfig([
                'frame.xTickLength' => 0, 'title.position' => 'down', 'title.margin' => 0,]);
            if ($size > 1) {
                [$fig, $axes] = $plt->subplots(4, 4);
            } else {
                [$fig, $axes] = $plt->subplots(1, 2);
            }


            foreach ($predicts as $i => $predict) {

                $axes[$i * 2]->imshow($images_j[$i],
                    null, null, null, $origin = 'upper');
                $axes[$i * 2]->setFrame(false);
                $label = $labels_j[$i];
                $axes[$i * 2]->setTitle($classnames[$label] . "(" . array_search($classnames[$label], $og_classnames) . ")");

                $axes[$i * 2 + 1]->bar($mo->arange(16), $predict);
            }

            return $plt;
        } else {
            $predicted_image = [];
            foreach ($predicts as $i => $predict) {

                $label = $labels_j[$i];
                $predicted_image["face"] = "face".array_search(max($predict->toArray()),$predict->toArray());
                $predicted_image["accuracy"] = max($predict->toArray());

            }

            $end_time = microtime(true);

            return ['predict' => $predicted_image, 'time' => $end_time-$start_time];
        }



    }
}




