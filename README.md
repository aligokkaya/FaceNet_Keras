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


```
pip install pickle

```
see:[Pickle](https://www.geeksforgeeks.org/pickle-python-object-serialization/?ref=lbp).

# Run

1-data-prepare.py
Output:
![SS](https://user-images.githubusercontent.com/43973787/97847563-dd017d80-1d00-11eb-9b87-975f45ac930d.PNG)




2-camera.py and image.py
Output:
![ss3](https://user-images.githubusercontent.com/43973787/97849107-2226af00-1d03-11eb-8e0d-d8f7b439d2e8.png)






