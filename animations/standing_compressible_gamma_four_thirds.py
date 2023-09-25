import numpy as np

# Not done yet...


def drhoc_over_rhoc(x, a):
    """
    Compressible, standing wave with γ=4/3
    initial velocity.
    Solution assumes real kappa
    """
    assert A_rho == 0
    psi = kappa*np.log(a/ai)
    res = A_u*(a/ai)**(-1/4)*OmegaS*np.sqrt(ai)/kappa*np.sin(psi)*np.sin(k*x)
    return res


def du_over_vs(x, a):
    """
    Compressible, standing wave with γ=4/3
    initial velocity.
    Solution assumes real kappa
    """
    assert A_rho == 0
    psi = kappa*np.log(a/ai)

    res = A_u*(a/ai)**(-3/4)*(np.cos(psi) - np.sin(psi)/(4*kappa))*np.cos(k*x)
    return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from comoving_mhd_waves import ScipyComovingMagnetosonicWave

    # Animation of standing compressible wave.
    # Parameters as in Fig. 2 in the paper

    A_u = 1
    A_rho = 0

    mhd = True
    selfgravity = True

    k = 2*np.pi
    H0 = 1
    Vs = 1/5
    Va = 1/2
    Vg = 1/4

    gamma = 4/3

    ai = 1/128

    if not mhd:
        Va = 0
    if not selfgravity:
        Vg = 0

    OmegaS = k*Vs/H0
    OmegaA = k*Va/H0
    OmegaG = k*Vg/H0

    sigma = np.sqrt(OmegaS**2 + OmegaA**2 - OmegaG**2, dtype=np.complex128)
    kappa = np.sqrt(sigma**2 - 1/16, dtype=np.complex128).real

    sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai,
                                            A_u, A_rho)

    x = np.linspace(0, 1, 256)

    plt.figure(2)
    plt.clf()

    fig, axes = plt.subplots(num=2, nrows=2, ncols=2, sharex=True,
                             sharey='row')

    line1, = axes[0, 0].plot(x, drhoc_over_rhoc(x, ai), label='Analytic')
    line2, = axes[0, 1].plot(x, drhoc_over_rhoc(x, ai))
    line3, = axes[1, 0].plot(x, du_over_vs(x, ai))
    line4, = axes[1, 1].plot(x, du_over_vs(x, ai))

    line5, = axes[0, 0].plot(x, (sci_sol.delta_rhoc_over_rhoc(ai)
                                 * np.exp(1j*k*x)).real, '--', label='Scipy')
    line6, = axes[0, 1].plot(x, (sci_sol.delta_rhoc_over_rhoc(ai)
                                 * np.exp(1j*k*x)).real, '--')
    line7, = axes[1, 0].plot(x, (sci_sol.delta_u_over_vs(ai)
                                 * np.exp(1j*k*x)).real, '--')
    line8, = axes[1, 1].plot(x, (sci_sol.delta_u_over_vs(ai)
                                 * np.exp(1j*k*x)).real, '--')

    axes[0, 0].legend(frameon=False)

    axes[0, 0].plot(x, drhoc_over_rhoc(x, ai), 'k:')
    axes[0, 1].plot(x, drhoc_over_rhoc(x, ai), 'k:')
    axes[1, 0].plot(x, du_over_vs(x, ai), 'k:')
    axes[1, 1].plot(x, du_over_vs(x, ai), 'k:')

    for ii in range(2):
        axes[1, ii].set_xlabel(r'$x/L$')

    axes[0, 0].set_ylabel(r'$\delta \rho_\mathrm{c}/\rho_\mathrm{c}$')
    axes[0, 1].set_ylabel(r'$(a/a_\mathrm{i})^{1/4}\delta \rho_\mathrm{c}/\rho_\mathrm{c}$')
    axes[1, 0].set_ylabel(r'$\delta u/\mathcal{V}_\mathrm{s}$')
    axes[1, 1].set_ylabel(r'$(a/a_\mathrm{i})^{3/4}\delta u/\mathcal{V}_\mathrm{s}$')

    avec = np.logspace(np.log10(ai), 0, 200)
    ymax = 2*np.max(-sci_sol.delta_rhoc_over_rhoc(avec).imag)
    axes[0, 0].set_ylim(-ymax, ymax)

    for a in avec:

        # Analytic solution
        line1.set_ydata(drhoc_over_rhoc(x, a))
        line2.set_ydata((a/ai)**(1/4)*drhoc_over_rhoc(x, a))
        line3.set_ydata(du_over_vs(x, a))
        line4.set_ydata((a/ai)**(3/4)*du_over_vs(x, a))

        # Scipy solution
        line5.set_ydata((sci_sol.delta_rhoc_over_rhoc(a)*np.exp(1j*k*x)).real)
        line6.set_ydata((a/ai)**(1/4)*(sci_sol.delta_rhoc_over_rhoc(a)
                                       * np.exp(1j*k*x)).real)
        line7.set_ydata((sci_sol.delta_u_over_vs(a)*np.exp(1j*k*x)).real)
        line8.set_ydata((a/ai)**(3/4)*(sci_sol.delta_u_over_vs(a)
                                       * np.exp(1j*k*x)).real)

        fig.suptitle(r'$\ln(a/ai) = ' + '{:1.2f}'.format(np.log(a/ai)) + '$')
        plt.pause(1e-2)

    plt.show()
