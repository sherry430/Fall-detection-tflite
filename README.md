# FD-CNN(Fall Detection Convolutional Neural Networks)
the paper[https://ieeexplore.ieee.org/document/8662651]


## Net construct


![image_1](md/AFD_cnn.png)



## Net performance
![image_2](md/性能.png)


## How to train and test
    python ./src/cnn.py

## How to transform model to tflite
    python ./ckpt2pb.py
    python ./pb2tflite.py

## Dataset

- SisFall http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/

