## Reverse Auto-Diff

This library contains the code to support a automatic differentiation.
The code makes use of `templates` to make it agnostic to the underlying data that it works on. However as per the implementation of the `DataType` few things may need to be tweaked to get it all working.

The aim of the library is to be able to manipulate matrices and tensors. Ideally it should be able to support external libraries like
`Eigen` to make things easier.

### How it works?
This library has been heavily influenced by this great [blog](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation).

#### Node
One can imagime the whole computation as a one big graph from start to end. Each node captures the information of what is transformed into what. It also in some sense keeps track of how (which helps to get the gradients).

#### Tape
It is the actually structure that holds the graph.

#### Tensor
This is the structure that holds the data and applies and computes the transforms and informs tape by adding the nodes to the tape.

Note: A more updated documentation will be updated in some time.

#### Examples
The test file holds some examples that can get you started.
Sadly as of now the library can only work it doubles and float.
However it can be easily (that's what I believe) be refactored to work
with Eigen or any other library.



