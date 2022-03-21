import numpy as np


def compare(show, a, an_sol, sci_sol):
    """
    Do the testing and plot if requested
    """
    np.testing.assert_allclose(an_sol.delta_rhoc_over_rhoc(a).real,
                               sci_sol.delta_rhoc_over_rhoc(a).real, atol=1e-5)
    np.testing.assert_allclose(an_sol.delta_rhoc_over_rhoc(a).imag,
                               sci_sol.delta_rhoc_over_rhoc(a).imag, atol=1e-5)
    np.testing.assert_allclose(an_sol.delta_u_over_vs(a).real,
                               sci_sol.delta_u_over_vs(a).real, atol=1e-5)
    np.testing.assert_allclose(an_sol.delta_u_over_vs(a).imag,
                               sci_sol.delta_u_over_vs(a).imag, atol=1e-5)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=2, ncols=2, sharex=True)

        axes[0, 0].semilogx(a, an_sol.delta_rhoc_over_rhoc(a).real, label='Analytic')
        axes[0, 1].semilogx(a, an_sol.delta_rhoc_over_rhoc(a).imag)
        axes[1, 0].semilogx(a, an_sol.delta_u_over_vs(a).real)
        axes[1, 1].semilogx(a, an_sol.delta_u_over_vs(a).imag)
        axes[0, 0].set_title(r'$\mathrm{Re}(\delta \rho_\mathrm{c}/\rho_\mathrm{c})$')
        axes[0, 1].set_title(r'$\mathrm{Im}(\delta \rho_\mathrm{c}/\rho_\mathrm{c})$')
        axes[1, 0].set_title(r'$\mathrm{Re}(\delta u/\mathcal{V}_\mathrm{A})$')
        axes[1, 1].set_title(r'$\mathrm{Im}(\delta u/\mathcal{V}_\mathrm{A})$')

        axes[0, 0].semilogx(a, sci_sol.delta_rhoc_over_rhoc(a).real, '--', label='Scipy')
        axes[0, 1].semilogx(a, sci_sol.delta_rhoc_over_rhoc(a).imag, '--')
        axes[1, 0].semilogx(a, sci_sol.delta_u_over_vs(a).real, '--')
        axes[1, 1].semilogx(a, sci_sol.delta_u_over_vs(a).imag, '--')
        axes[0, 0].legend(frameon=False)
        nums = '{:1.2f},{:1.2f},{:1.2f}'.format(an_sol.OmegaS/np.pi, an_sol.OmegaA/np.pi, an_sol.OmegaG/np.pi)
        title = r',\, (\Omega_\mathrm{s},\,\Omega_\mathrm{A},\,\Omega_\mathrm{g})/\pi = ' + nums + '$'
        title = r'$\gamma=' + '{:1.2f}'.format(an_sol.gamma) + title
        fig.suptitle(title)

        plt.show()


def test_isothermal_compressible_wave_standing(show=False):
    from comoving_mhd_waves import AnalyticComovingMagnetosonicWave
    from comoving_mhd_waves import ScipyComovingMagnetosonicWave

    ai = 1/128
    A_u = 1
    A_rho = 0.2

    H0 = 1
    k = 2*np.pi

    a = np.logspace(np.log10(ai), 0, 300)

    Vs = 1.0
    gamma = 1
    for Va in [0, 1/2]:
        for Vg in [0, 1/4]:
            an_sol = AnalyticComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
            sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
            compare(show, a, an_sol, sci_sol)


def test_gamma_four_thirds_compressible_wave_standing(show=False, ii=0):
    import numpy as np
    from comoving_mhd_waves import AnalyticComovingMagnetosonicWave
    from comoving_mhd_waves import ScipyComovingMagnetosonicWave

    ai = 1/128
    A_u = 1
    A_rho = 0.2

    H0 = 1
    k = 2*np.pi

    a = np.logspace(np.log10(ai), 0, 300)

    Vs = 1/5
    gamma = 4/3
    for Va in [0, 1/2]:
        for Vg in [0, 1/4]:
            an_sol = AnalyticComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
            sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
            compare(show, a, an_sol, sci_sol)

    Vs = 1/4 * H0/k
    Va = 0
    Vg = 0

    an_sol = AnalyticComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
    sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
    compare(show, a, an_sol, sci_sol)


def test_gamma_five_thirds_compressible_wave_standing(show=False, ii=0):
    import numpy as np
    from comoving_mhd_waves import AnalyticComovingMagnetosonicWave
    from comoving_mhd_waves import ScipyComovingMagnetosonicWave

    ai = 1/128
    A_u = 1
    A_rho = 0.2

    H0 = 1
    k = 2*np.pi

    a = np.logspace(np.log10(ai), 0, 300)

    Vs = 1/10
    gamma = 5/3
    for Va in [0, 1/2]:
        for Vg in [0, 1/4]:
            an_sol = AnalyticComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
            sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
            compare(show, a, an_sol, sci_sol)


def test_gamma_nonstandard_compressible_wave_standing(show=False, ii=0):
    import numpy as np
    from comoving_mhd_waves import AnalyticComovingMagnetosonicWave
    from comoving_mhd_waves import ScipyComovingMagnetosonicWave

    ai = 1/128
    A_u = 1
    A_rho = 0.2

    H0 = 1
    k = 2*np.pi

    a = np.logspace(np.log10(ai), 0, 300)

    Vs = 1/2
    gamma = 1.2
    """
    The Bessel function solution does not work so well for imaginary ν
    and γ≠1 and 5/3. The problem appears to be with the mpmath bessel
    functions. It might be worthwhile to see if Mathematica behaves better
    and/or simply use the scipy solution.
    One possibility might be to completely rewrite the solution as in the
    paper
    BESSEL FUNCTIONS OF PURELY IMAGINARY ORDER, WITH AN APPLICATION TO
    SECOND-ORDER LINEAR DIFFERENTIAL EQUATIONS HAVING A LARGE PARAMETER
    T. M. DUNSTER
    """
    for Va in [0, 1/8]:
        for Vg in [0, 1/4]:
            an_sol = AnalyticComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
            sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
            compare(show, a, an_sol, sci_sol)


if __name__ == '__main__':
    test_isothermal_compressible_wave_standing(show=True)
    test_gamma_four_thirds_compressible_wave_standing(show=True)
    test_gamma_five_thirds_compressible_wave_standing(show=True)
    test_gamma_nonstandard_compressible_wave_standing(show=True)
