
def test_compressible_wave_standing(show=False):
    import numpy as np
    from comoving_mhd_waves import AnalyticComovingMagnetosonicWave
    from comoving_mhd_waves import ScipyComovingMagnetosonicWave

    ai = 1/128
    A_u = 1
    A_rho = 0.2

    gamma = 4/3 + 1e-8
    rhoc = 1
    Bc = 0.2
    H0 = 1
    k = 2*np.pi
    pc_0 = 1
    G = 0.2

    # 'Velocities'
    Vs = np.sqrt(gamma*pc_0/rhoc)
    Va = Bc/np.sqrt(rhoc)
    Vg = np.sqrt(4*np.pi*G*rhoc)/k

    an_sol = AnalyticComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
    sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)

    a = np.linspace(ai, 1, 5000)

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

        axes[0, 0].semilogx(a, an_sol.delta_rhoc_over_rhoc(a).real, '--', label='Scipy')
        axes[0, 1].semilogx(a, an_sol.delta_rhoc_over_rhoc(a).imag, '--')
        axes[1, 0].semilogx(a, an_sol.delta_u_over_vs(a).real, '--')
        axes[1, 1].semilogx(a, an_sol.delta_u_over_vs(a).imag, '--')
        axes[0, 0].legend(frameon=False)
        plt.show()

    # Test that everything works
    np.testing.assert_allclose(an_sol.delta_rhoc_over_rhoc(a).real,
                               sci_sol.delta_rhoc_over_rhoc(a).real, atol=1e-6)
    np.testing.assert_allclose(an_sol.delta_rhoc_over_rhoc(a).imag,
                               sci_sol.delta_rhoc_over_rhoc(a).imag, atol=1e-6)
    np.testing.assert_allclose(an_sol.delta_u_over_vs(a).real,
                               sci_sol.delta_u_over_vs(a).real, atol=1e-6)
    np.testing.assert_allclose(an_sol.delta_u_over_vs(a).imag,
                               sci_sol.delta_u_over_vs(a).imag, atol=1e-6)


if __name__ == '__main__':
    test_compressible_wave_standing(show=True)
