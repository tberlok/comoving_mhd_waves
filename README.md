[![CircleCI](https://dl.circleci.com/status-badge/img/gh/tberlok/comoving_mhd_waves/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/tberlok/comoving_mhd_waves/tree/main)

# Comoving MHD waves

This Github repository contains supplementary Python scripts for the MNRAS
paper:

*Hydromagnetic waves in an expanding universe – cosmological MHD code tests
using analytic solutions*, Berlok T., 2022, MNRAS, 515, 3492. doi:10.1093/mnras/stac1882 [ADS-link](https://ui.adsabs.harvard.edu/abs/2022MNRAS.515.3492B/abstract).

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


