# HQNN Classification

Trying to recreate the results detailed in this paper: https://arxiv.org/pdf/2304.09224v2

# Files

- `HQNN-Parallel-local.ipynb`: A manual implementation of the HQNN algorithm, with a custom training loop and optimization step. This is a more manual approach which lets me control the training process and types more closely.
- `HQNN-Parallel-ionq.ipynb`: My attempt to run the model from the local notebook on IonQ machines. This includes:
  - The IonQ ideal simulator
  - IonQ simulator with "aria-1" noise model
  - IonQ simulator with "forte-1" noise model
  - IonQ's "aria-1" quantum computer
  - IonQ's "forte-1" quantum computer
  - IonQ's "forte-enterprise-1" quantum computer
    Things to consider with this:
  - Inability to directly calculate expectation values - must sample
  - Much longer training times, due to waiting in queues and communication time between client and job-managing servers

## HQNN-Parallel-manual.ipynb

This notebook accurately implements the HQNN-Parallel algorithm as described in the paper. Using the local `lightning.qubit` local simulator, it achieves a high accuracy score on the MNIST dataset.
