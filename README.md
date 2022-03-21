[![CircleCI](https://circleci.com/gh/tberlok/comoving_mhd_waves/tree/main.svg?style=svg&circle-token=33bce19b6fe69af562d7c2519ea4d6ab12958290)](https://circleci.com/gh/tberlok/comoving_mhd_waves/tree/main)

# Comoving MHD waves

This Github repository contains supplementary Python scripts for the MNRAS
submitted paper:

*Hydromagnetic waves in an expanding universe – cosmological MHD code tests
using analytic solutions* [arxiv]().

# Installation

I assume you have Python 3 installed. If so, all requirements are simply
installed by running the following command

```
$ pip install -r requirements.txt
```
at the top-level directory.

# Testing

Before using the code, the tests should be run to make sure that they are
working. From the top-level directory
```
$ pytest tests/
```


