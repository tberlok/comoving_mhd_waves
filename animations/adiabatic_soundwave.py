import numpy as np
from numpy import sqrt, cos, sin


def drhoc_over_rhoc(x, a):
    """
    Adiabatic sound wave with initial
    velocity perturbation
    """
    assert A_rho == 0
    phi = 2*OmegaS*(1/sqrt(ai) - 1/sqrt(a))

    drhoc_over_rhoc = A_u*ai*sin(phi)*sin(k*x)
    return drhoc_over_rhoc


def du_over_vs(x, a):
    """
    Adiabatic sound wave with initial
    velocity perturbation
    """
    assert A_rho == 0
    phi = 2*OmegaS*(1/sqrt(ai) - 1/sqrt(a))
    du_over_Vs = A_u*ai/a*cos(phi)*cos(k*x)
    return du_over_Vs


if __name__ == '__main__':
    from comoving_mhd_waves import ScipyComovingMagnetosonicWave
    import matplotlib.pyplot as plt

    # Animation of adiabatic sound wave.
    # Parameters as in Fig. 4 in the paper

    k = 2*np.pi
    H0 = 1
    Vs = 1/10
    Va = Vg = 0
    gamma = 5/3

    OmegaS = k*Vs/H0
    A_u = 1
    A_rho = 0
    ai = 1/128

    sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai,
                                            A_u, A_rho)

    x = np.linspace(0, 1, 256)

    plt.figure(2)
    plt.clf()

    fig, axes = plt.subplots(num=2, nrows=2, sharex=True)

    line1, = axes[0].plot(x, drhoc_over_rhoc(x, ai), label='Analytic')
    line2, = axes[1].plot(x, du_over_vs(x, ai))
    line3, = axes[0].plot(x, drhoc_over_rhoc(x, ai), '--', label='Scipy')
    line4, = axes[1].plot(x, du_over_vs(x, ai), '--')
    axes[0].plot(x, drhoc_over_rhoc(x, ai), 'k:')
    axes[1].plot(x, du_over_vs(x, ai), 'k:')

    axes[1].set_xlabel(r'$x/L$')
    axes[0].legend(frameon=False)

    axes[0].set_ylabel(r'$\delta \rho_\mathrm{c}/\rho_\mathrm{c}$')
    axes[1].set_ylabel(r'$\delta u/\mathcal{V}_\mathrm{s}$')

    avec = np.logspace(np.log10(ai), 0, 200)
    ymax = np.max(-sci_sol.delta_rhoc_over_rhoc(avec).imag)
    axes[0].set_ylim(-ymax, ymax)

    for a in avec:

        line1.set_ydata(drhoc_over_rhoc(x, a))
        line2.set_ydata(du_over_vs(x, a))
        line3.set_ydata((sci_sol.delta_rhoc_over_rhoc(a)*np.exp(1j*k*x)).real)
        line4.set_ydata((sci_sol.delta_u_over_vs(a)*np.exp(1j*k*x)).real)
        fig.suptitle(r'$\ln(a/ai) = ' + '{:1.2f}'.format(np.log(a/ai)) + '$')
        plt.pause(1e-2)

    plt.show()
