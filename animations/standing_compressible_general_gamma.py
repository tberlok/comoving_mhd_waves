import numpy as np
from comoving_mhd_waves import J, Y, J_p, Y_p


def F(a):
    """
    This is mathcal{F} as defined in the paper.
    """
    z = OmegaS*a**s/np.abs(s)
    return a**(-1/4)*J(nu, z)


def G(a):
    """
    This is mathcal{G} as defined in the paper.
    """
    z = OmegaS*a**s/np.abs(s)
    return a**(-1/4)*Y(nu, z)


def Fp(a):
    """
    The derivative of F
    """
    z = OmegaS*a**s/np.abs(s)
    sgn_s = np.sign(s)
    y = -1/4*J(nu, z)
    y += OmegaS*sgn_s * a**s * J_p(nu, z)
    return y*a**(-5/4)


def Gp(a):
    """
    The derivative of G
    """
    z = OmegaS*a**s/np.abs(s)
    sgn_s = np.sign(s)
    y = -1/4*Y(nu, z)
    y += OmegaS*sgn_s * a**s * Y_p(nu, z)
    return y*a**(-5/4)


def drhoc_over_rhoc(x, a):
    """
    Compressible wave with initial
    velocity perturbation
    """
    assert A_rho == 0
    assert gamma != 4/3

    res = A_u*1j*OmegaS*np.pi*ai/(2*s)*(G(ai)*F(a) - F(ai)*G(a))
    return (res*np.exp(1j*k*x)).real


def du_over_vs(x, a):
    """
    Compressible wave with initial
    velocity perturbation
    """
    assert A_rho == 0
    assert gamma != 4/3

    res = -A_u*np.sqrt(a)*np.pi*ai/(2*s)*(G(ai)*Fp(a) - F(ai)*Gp(a))
    return (res*np.exp(1j*k*x)).real


if __name__ == '__main__':
    from comoving_mhd_waves import ScipyComovingMagnetosonicWave
    import matplotlib.pyplot as plt

    # Animation of wave with γ=7/5. Not included in paper.
    # We have here set the parameters to be gravitationally unstable.
    # Note that the Bessel functions have issues in some cases,
    # see comment in compressible_analytic.py

    A_u = 1
    A_rho = 0

    mhd = False
    selfgravity = True

    k = 2*np.pi
    H0 = 1
    Vs = 1/3
    Va = 1/4
    Vg = 1/2

    gamma = 7/5

    ai = 1/128

    if not mhd:
        Va = 0
    if not selfgravity:
        Vg = 0

    OmegaS = k*Vs/H0
    OmegaA = k*Va/H0
    OmegaG = k*Vg/H0

    s = (4 - 3*gamma)/2
    nu = np.sqrt(1 - 16*(OmegaA**2 - OmegaG**2),
                 dtype=np.complex128)/(4*np.abs(s))

    sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai,
                                            A_u, A_rho)

    x = np.linspace(0, 1, 256)

    plt.figure(2)
    plt.clf()

    fig, axes = plt.subplots(num=2, nrows=2, sharex=True)

    line1, = axes[0].plot(x, drhoc_over_rhoc(x, ai))
    line2, = axes[1].plot(x, du_over_vs(x, ai))
    line3, = axes[0].plot(x, drhoc_over_rhoc(x, ai), '--')
    line4, = axes[1].plot(x, du_over_vs(x, ai), '--')
    axes[0].plot(x, drhoc_over_rhoc(x, ai), 'k:')
    axes[1].plot(x, du_over_vs(x, ai), 'k:')

    axes[1].set_xlabel(r'$x/L$')

    axes[0].set_ylabel(r'$\delta \rho_\mathrm{c}/\rho_\mathrm{c}$')
    axes[1].set_ylabel(r'$\delta u/\mathcal{V}_\mathrm{s}$')

    avec = np.logspace(np.log10(ai), 0, 200)
    ymax = np.max(-sci_sol.delta_rhoc_over_rhoc(avec).imag)
    axes[0].set_ylim(-ymax, ymax)
    ymax = np.max(sci_sol.delta_u_over_vs(avec).real)
    axes[1].set_ylim(-ymax, ymax)

    for a in avec:

        line1.set_ydata(drhoc_over_rhoc(x, a))
        line2.set_ydata(du_over_vs(x, a))
        line3.set_ydata((sci_sol.delta_rhoc_over_rhoc(a)*np.exp(1j*k*x)).real)
        line4.set_ydata((sci_sol.delta_u_over_vs(a)*np.exp(1j*k*x)).real)
        fig.suptitle(r'$\ln(a/ai) = ' + '{:1.2f}'.format(np.log(a/ai)) + '$')
        plt.pause(1e-2)

    plt.show()
