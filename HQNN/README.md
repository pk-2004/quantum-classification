# HQNN Classification

Trying to recreate the results detailed in this paper: https://arxiv.org/pdf/2304.09224v2

# Files

- `HQNN-Parallel-tf.ipynb`: A Tensorflow implementation of the HQNN algorithm. I created a custom Tensorflow model with custom Tensorflow layers for each part of the model in the paper. I am trying to train it on the MNIST dataset to start, but there is a type issue between float32 and float64 in the gradient calculations.
- `HQNN-Parallel-manual.ipynb`: A manual implementation of the HQNN algorithm, with a custom training loop and optimization step. This is a more manual approach which lets me control the training process and types more closely.
