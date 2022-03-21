import numpy as np
import mpmath as mp
mp.dps = 20

"""
    Here we define functions for returning the Bessel functions J and Y
    and their derivatives, Jp and Yp.

    These functions use mpmath if the order is imaginary and
    scipy if the order is real.
"""


def convert_mp_to_np(y_mp):
    y_np = mp.nstr(y_mp, 17, min_fixed=0, max_fixed=0).replace(' ', '')
    return np.complex(y_np)


def bessel_j(nu, x):
    from mpmath import besselj
    y = besselj(nu, x)
    return convert_mp_to_np(y)


def bessel_j_prime(nu, x):
    from mpmath import besselj
    y = besselj(nu, x, 1)
    return convert_mp_to_np(y)


def bessel_y(nu, x):
    from mpmath import bessely
    y = bessely(nu, x)
    return convert_mp_to_np(y)


def bessel_y_prime(nu, x):
    from mpmath import bessely
    y = bessely(nu, x, 1)
    return convert_mp_to_np(y)


# Vectorized version of the function
besselj = np.vectorize(bessel_j)
besselj_p = np.vectorize(bessel_j_prime)
bessely = np.vectorize(bessel_y)
bessely_p = np.vectorize(bessel_y_prime)


def J(nu, z):
    # Real order
    if nu.imag == 0:
        from scipy.special import jv
        return jv(nu.real, z)
    else:
        return besselj(nu, z)


def J_p(nu, z):
    # Real order
    if nu.imag == 0:
        from scipy.special import jvp
        return jvp(nu.real, z)
    else:
        return besselj_p(nu, z)


def Y(nu, z):
    # Real order
    if nu.imag == 0:
        from scipy.special import yv
        return yv(nu.real, z)
    else:
        return bessely(nu, z)


def Y_p(nu, z):

    # Real order
    if nu.imag == 0:
        from scipy.special import yvp
        return yvp(nu.real, z, 1)
    else:
        return bessely_p(nu, z)
