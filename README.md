
[![Build Status](https://travis-ci.org/felixquinton/tyssue-taylor.svg?branch=master)](https://travis-ci.org/felixquinton/tyssue-taylor)

# tyssue-taylor - From experimental data to a tyssue model

This package uses the [`tyssue`](https://tyssue.readthedocs.io) library to adjust a vertex model to experimental microscopy data.

## Dependencies

- tifffile
- swig
- opencv
- tyssue > 0.5
- sympy
- tensorflow
- stardist
- python >= 3.6


## Installation

- clone this repository from github:

```sh
git clone https://github.com/glyg/tyssue-taylor.git
```

- In the cloned repository create a virtual environment with conda:
```sh
cd tyssue-taylor
conda env create -f environment.yml
```

- Create the environment and install the package
```bash
conda activate taylor
python setup.py install
```

See [This notebook](notebooks/SyntheticMesh.ipynb) for an example of how to use the library.

## Licence

This project is distributed under the terms of the [Modzilla Public Licence](https://www.mozilla.org/en-US/MPL/2.0/).

## Use

TODO
