
[![Build Status](https://api.travis-ci.org/felixquinton/tyssue-taylor.svg?branch=mastersta)](https://travis-ci.org/felixquinton/tyssue-taylor)


# tyssue-taylor - From experimental data to a tyssue model

This package uses the [`tyssue`](https://tyssue.readthedocs.io) library to adjust a vertex model to experimental microscopy data.

## Dependencies

- python >= 3.6
- tyssue >= 0.2
- python-opencv

TODO: complete this and put it in a requirements.txt

## Installation

- Install from source by cloning the git repository
- Then in the package directory, run
```bash
python setup.py install
```

See [This notebook](notebooks/SyntheticMesh.ipynb) for an example of how to use the library.


## Licence

This project is distributed under the terms of the [Modzilla Public Licence](https://www.mozilla.org/en-US/MPL/2.0/).

## Use

### Create an annular mesh from real data.

Using segmentation.segment2D.generate_ring_from_image, one can initialize a
mesh from a brightfield image and a CSV file from the nuclei_extraction CellProfiler
pipeline.
To obtain such a CSV file require to run nuclei_extraction on a DAPI image.
generate_ring_from_image creates an object of class AnnularSheet.

### Compute the linear tensions.

The adjusters module provide some tools to find the linear tensions of the
edges of an AnnularSheet object.
To do so, linear tensions will be considered as parameters of an optimization
problem.

There are two optimization problems :

- Minimizing the distance between the experimental organoïd and the theoritical
organoïd, without constraint. This problem can be solved with Trust Region Function method or Levenberg-Marquardt method. To refine the results from this problem,
one may introduce a regularization module. It add the term $(L_i-L_{i+1})^2$ for each edge on the regularized ring.
- Minimizing the energy of the experimental organoïd under constraint that the
distance between the experimental organoïd and the theoritical organoïd is less
than a given threshold. The initial point for this method is computed by solving the problem described above and its objective value gives the distance threshold. This problem can be solved with Preconditionned Sequential Quadratic Programming.
