# Evaluating a machine learning function with functional encryption - example

This repository demonstrates how the  [SGP]() functional encryption (FE)
scheme for evaluation of quadratic multi-variate polynomials can be used
to evaluate a machine learning function on encrypted data. 

Specifically, the test data shipped with this repository provides all that
we need to predict which number is hiding behind the image image of a handwritten
   digit encoded as a matrix of grayscale levels saved in `testdata/mat_valid.txt`
   also seen below:

```
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 0 1 3 2 1 0 0 0 0
0 0 0 0 0 0 0 0 0 0 1 1 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 1 1 2 2 2 0 0 0 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 1 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 2 2 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 2 2 2 2 2 2 2 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 2 2 2 2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
```

However, the goal of this example is to demonstrate that a
party can encrypt the input data, while another party can learn something
from the encrypted data (e.g. which number is represented on the image)
without actually knowing exactly what the image was.


The example uses an implementation of the scheme from the 
[GoFE library](https://github.com/fentec-project/gofe).

### Input data
The example assumes that we learned a function for recognizing numbers from
images of hand-written digits using TensorFlow, and saved the parameters to files
`mat_diag.txt` and `mat_proj.txt`, which we put in the `testdata` folder.
* mat_valid.txt holds a vector that needs to be encrypted
* mat_proj.txt holds the first set of parameters of the function deciding which digit is the image
* mat_diag.txt holds the second set of parameters of the function deciding which digit is the image

We used a Python script `mnist.py` to train the model. Feel free to continue with the
section [How to run the example](#how-to-run-the-example) if you wish to run the 
example without re-running the training script.

#### Generating training data
If you wish to re-train the model (and generate new input data for the example),
 you will need to:
 
1. Install the dependencies:
````bash
$ pip install tensorflow numpy
````
2. Navigate to the root of this repository and run the
python training script:
````bash
$ cd $GOPATH/src/github.com/fentec-project/fe-ml-example
$ python mnist.py
````

## How to run the example
1. Build the example by running:
    ````bash
    $ go get github.com/fentec-project/fe-ml-example
    ````
2. This will produce the `fe-ml-example` executable in your `$GOPATH/bin`.
If you have `$GOPATH/bin` in your `PATH` environment variable, you
will be able to run the example by running command `fe-ml-example` from the 
root of this repository. 

    Otherwise just call:
    ```bash
    $ $GOPATH/bin/fe-ml-example
    ```
    The example will output the predicted number behind the image 
    (encoded as a matrix in `testdata/mat_valid.txt`):
    ````bash
    prediction vector: [-99073763 -149651697 -114628671 83732640 -387336224 130856071 -302672454 -126041027 -121102209 -111101930]
    the model predicts that the number is 5
    ````