import numpy as np


def dBc_over_Bc(x, a):
    """
    Standing Alfvén wave with velocity perturbation only
    """
    assert A_B == 0
    psi = kappa*np.log(a/ai)
    res = -(a/ai)**(-1/4)*np.sqrt(ai)*OmegaA/kappa*np.sin(psi)*np.sin(k*x)
    return res


def du_over_va(x, a):
    """
    Standing Alfvén wave with velocity perturbation only
    """
    assert A_B == 0
    psi = kappa*np.log(a/ai)
    res = (a/ai)**(-3/4)*(np.cos(psi) - np.sin(psi)/(4*kappa))*np.cos(k*x)
    return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from comoving_mhd_waves import ScipyComovingAlfvenWave

    # Animation of standing Alvén wave.
    # Parameters as in Fig. 1 in the paper

    k = 2*np.pi
    H0 = 1
    Va = 1/2
    OmegaA = k*Va/H0

    kappa = np.sqrt(OmegaA**2 - 1/16)
    ai = 1/128

    A_u = 1
    A_B = 0

    sci_sol = ScipyComovingAlfvenWave(OmegaA, ai, A_u, A_B)

    x = np.linspace(0, 1, 256)

    plt.figure(2)
    plt.clf()
    fig, axes = plt.subplots(num=2, nrows=2, ncols=2, sharex=True,
                             sharey='row')

    line1, = axes[0, 0].plot(x, dBc_over_Bc(x, ai), label='Analytic')
    line2, = axes[0, 1].plot(x, dBc_over_Bc(x, ai))
    line3, = axes[1, 0].plot(x, du_over_va(x, ai))
    line4, = axes[1, 1].plot(x, du_over_va(x, ai))

    line5, = axes[0, 0].plot(x, (sci_sol.delta_Bc_over_Bc(ai)
                                 * np.exp(1j*k*x)).real, '--', label='Scipy')
    line6, = axes[0, 1].plot(x, (sci_sol.delta_Bc_over_Bc(ai)
                                 * np.exp(1j*k*x)).real, '--')
    line7, = axes[1, 0].plot(x, (sci_sol.delta_u_over_va(ai)
                                 * np.exp(1j*k*x)).real, '--')
    line8, = axes[1, 1].plot(x, (sci_sol.delta_u_over_va(ai)
                                 * np.exp(1j*k*x)).real, '--')

    axes[0, 0].legend(frameon=False)

    axes[0, 0].plot(x, dBc_over_Bc(x, ai), 'k:')
    axes[0, 1].plot(x, dBc_over_Bc(x, ai), 'k:')
    axes[1, 0].plot(x, du_over_va(x, ai), 'k:')
    axes[1, 1].plot(x, du_over_va(x, ai), 'k:')

    for ii in range(2):
        axes[1, ii].set_xlabel(r'$x/L$')

    axes[0, 0].set_ylabel(r'$\delta B_\mathrm{c}/B_\mathrm{c}$')
    axes[0, 1].set_ylabel(r'$(a/a_\mathrm{i})^{1/4}\delta B_\mathrm{c}/B_\mathrm{c}$')
    axes[1, 0].set_ylabel(r'$\delta u/\mathcal{V}_\mathrm{A}$')
    axes[1, 1].set_ylabel(r'$(a/a_\mathrm{i})^{3/4}\delta u/\mathcal{V}_\mathrm{A}$')

    avec = np.logspace(np.log10(ai), 0, 200)
    ymax = 2*np.max(-sci_sol.delta_Bc_over_Bc(avec).imag)
    axes[0, 0].set_ylim(-ymax, ymax)

    for a in avec:

        # Analytic solution
        line1.set_ydata(dBc_over_Bc(x, a))
        line2.set_ydata((a/ai)**(1/4)*dBc_over_Bc(x, a))
        line3.set_ydata(du_over_va(x, a))
        line4.set_ydata((a/ai)**(3/4)*du_over_va(x, a))

        # Scipy solution
        line5.set_ydata((sci_sol.delta_Bc_over_Bc(a)*np.exp(1j*k*x)).real)
        line6.set_ydata((a/ai)**(1/4)*(sci_sol.delta_Bc_over_Bc(a)
                                       * np.exp(1j*k*x)).real)
        line7.set_ydata((sci_sol.delta_u_over_va(a)*np.exp(1j*k*x)).real)
        line8.set_ydata((a/ai)**(3/4)*(sci_sol.delta_u_over_va(a)
                                       * np.exp(1j*k*x)).real)

        fig.suptitle(r'$\ln(a/ai) = ' + '{:1.2f}'.format(np.log(a/ai)) + '$')
        plt.pause(1e-2)

    plt.show()
