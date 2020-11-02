# FaceNet_Keras
## Install

1-Install dependencies

```
pip install pickle
pip install mtcnn
pip install opencv-python
pip install numpy

```

2-Adding new persons (or images of existing persons) into a system
First, creating a directory for raw images of new persons as follow.

```
$ tree data/new_person

people
    ├──person-name
          ├── image-1.jpg
          ├── image-2.png
          ├── image-3.jpeg
           ... 
          ├── image-p.png

           ...

    ├──person-name
          ├── image-1.png
          ├── image-2.jpg
          ├── image-3.jpeg
          ...
          └── image-q.png
          
          
```
# Dowland
## Dowland FaceNet-Keras Model 
This site was built using [GitHub Pages](https://github.com/D2KLab/Face-Celebrity-Recognition/tree/master/model).
```
model
    ├──facenet_keras.h5
```


# Pickle
## What Pickle

Pickle is used for serializing and de-serializing Python object structures, also called marshalling or flattening. Serialization refers to the process of converting an object in memory to a byte stream that can be stored on disk or sent over a network. Later on, this character stream can then be retrieved and de-serialized back to a Python object. Pickling is not to be confused with compression! The former is the conversion of an object from one representation (data in Random Access Memory (RAM)) to another (text on disk), while the latter is the process of encoding data with fewer bits, in order to save disk space.

see:[Pickle](https://docs.python.org/3/library/pickle.html#:~:text=%E2%80%9CPickling%E2%80%9D%20is%20the%20process%20whereby,back%20into%20an%20object%20hierarchy.).








