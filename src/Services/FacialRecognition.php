<?php

namespace App\Services;

use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Interop\Polite\Math\Matrix\NDArray;


class FacialRecognition{
    public static function train(): float
    {
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
            '../src/Services/training',
            pattern: '@.*\\.jpg@',
            batch_size: 32,
            height: 64,
            width: 64,
            shuffle: true);


        [$training, $training_labels] = $dataset->loadData();
        [$testing, $testing_labels] = $dataset_testing->loadData();

        echo 'images: ' . implode(',', $training->shape()) . "\n";
        echo 'labels: ' . implode(',', $training_labels->shape()) . "\n";
        echo 'images: ' . implode(',', $testing->shape()) . "\n";
        echo 'labels: ' . implode(',', $testing_labels->shape()) . "\n";

        $classnames = $dataset->classnames();
        var_dump($classnames);

        $pltCfg = [
            'title.position' => 'down', 'title.margin' => 0,
        ];

        $plt = new Plot($pltCfg, $mo);
        $images = $training[[0, 24]];
        $labels = $training_labels[[0, 24]];

        //Plot is messed up, have to randomize images or rearrange the array
        [$fig, $axes] = $plt->subplots(5, 5);
        foreach ($images as $i => $image) {
            $axes[$i]->imshow($image,
                null, null, null, $origin = 'upper');
            $label = $labels[$i];
            $axes[$i]->setTitle($classnames[$label] . "($label)");
            $axes[$i]->setFrame(false);
        }
        //$plt->show();

        $f_train_img = $mo->scale(1.0 / 255.0, $mo->la()->astype($training, NDArray::float32));
        $f_val_img = $mo->scale(1.0 / 255.0, $mo->la()->astype($testing, NDArray::float32));
        $i_train_label = $mo->la()->astype($training_labels, NDArray::int32);
        $i_train_label = $mo->la()->onehot($i_train_label, 16);

        $i_val_label = $mo->la()->astype($testing_labels, NDArray::int32);
        $i_val_label = $mo->la()->onehot($i_val_label, 16);

        $inputShape = $training->shape();
        array_shift($inputShape);

        //Initialize the neural network
        $model = $nn->models()->Sequential([
                $nn->layers()->Conv2D(
                    filters: 32,
                    kernel_size: [5, 5],
                    strides: [1, 1],
                    input_shape: [64, 64, 3],
                    kernel_initializer: 'he_normal',
                    activation: 'relu'),
                $nn->layers()->MaxPooling2D(
                    pool_size: [2, 2],),
                $nn->layers()->Conv2D(
                    filters: 64,
                    kernel_size: [5, 5],
                    strides: [1, 1],
                    kernel_initializer: 'he_normal',
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

        $train_dataset = $nn->data->ImageDataGenerator($f_train_img,
            tests: $training_labels,
            batch_size: 64,
            shuffle: true,
            height_shift: 2,
            width_shift: 2,
            vertical_flip: true,
            horizontal_flip: true
        );


        $model->compile(
            loss: $nn->losses()->CategoricalCrossEntropy(),
            optimizer: 'adam',
        );
        $history = $model->fit($f_train_img, $i_train_label,
            epochs: 10,
            validation_data: [$f_val_img, $i_val_label]);


        $plt->setConfig([]);
        $plt->plot($mo->array($history['accuracy']), null, null, 'accuracy');
        $plt->plot($mo->array($history['val_accuracy']), null, null, 'val_accuracy');
        $plt->plot($mo->array($history['loss']), null, null, 'loss');
        $plt->plot($mo->array($history['val_loss']), null, null, 'val_loss');
        $plt->legend();
        $plt->title('face_recognition');
        //$plt->show();

        $acc = end($history['accuracy']);

        $images_j = $f_val_img[[0, 7]];

        $i_val_label = $mo->la()->astype($testing_labels, NDArray::int32);
        $labels_j = $i_val_label[[0, 7]];

        $predicts = $model->predict($images_j);
        var_dump($f_val_img);
        var_dump($labels_j);


        $plt->setConfig([
            'frame.xTickLength' => 0, 'title.position' => 'down', 'title.margin' => 0,]);
        [$fig, $axes] = $plt->subplots(4, 4);
        foreach ($predicts as $i => $predict) {
            $axes[$i * 2]->imshow($images_j[$i]->reshape($inputShape),
                null, null, null, $origin = 'upper');
            $axes[$i * 2]->setFrame(false);
            $label = $labels_j[$i];
            $axes[$i * 2]->setTitle($classnames[$label] . "($label)");
            $axes[$i * 2 + 1]->bar($mo->arange(16), $predict);
        }
        //$plt->show();

        return $acc;
    }
}



