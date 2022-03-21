
def test_alfven_wave_standing(show=False):
    import numpy as np
    from comoving_mhd_waves import AnalyticComovingAlfvenWave
    from comoving_mhd_waves import ScipyComovingAlfvenWave

    ai = 1/128
    A_u = 1
    A_B = np.sqrt(ai)

    OmegaA = np.pi

    for ii, OmegaA in enumerate([1/16, 0.26, np.pi, 2*np.pi]):
        an_sol = AnalyticComovingAlfvenWave(OmegaA, ai, A_u, A_B)
        sci_sol = ScipyComovingAlfvenWave(OmegaA, ai, A_u, A_B)

        a = np.linspace(ai, 1, 5000)

        if show:
            import matplotlib.pyplot as plt
            plt.figure(ii+1)
            plt.clf()
            fig, axes = plt.subplots(num=ii+1, nrows=2, ncols=2, sharex=True)
            axes[0, 0].semilogx(a, an_sol.delta_Bc_over_Bc(a).real, label='Analytic')
            axes[0, 1].semilogx(a, an_sol.delta_Bc_over_Bc(a).imag)
            axes[1, 0].semilogx(a, an_sol.delta_u_over_va(a).real)
            axes[1, 1].semilogx(a, an_sol.delta_u_over_va(a).imag)

            axes[0, 0].semilogx(a, sci_sol.delta_Bc_over_Bc(a).real, '--', label='Scipy')
            axes[0, 1].semilogx(a, sci_sol.delta_Bc_over_Bc(a).imag, '--')
            axes[1, 0].semilogx(a, sci_sol.delta_u_over_va(a).real, '--')
            axes[1, 1].semilogx(a, sci_sol.delta_u_over_va(a).imag, '--')
            axes[0, 0].legend(frameon=False)

            axes[0, 0].set_title(r'$\mathrm{Re}(\delta B_\mathrm{c}/B_\mathrm{c})$')
            axes[0, 1].set_title(r'$\mathrm{Im}(\delta B_\mathrm{c}/B_\mathrm{c})$')
            axes[1, 0].set_title(r'$\mathrm{Re}(\delta u/\mathcal{V}_\mathrm{A})$')
            axes[1, 1].set_title(r'$\mathrm{Im}(\delta u/\mathcal{V}_\mathrm{A})$')

        # Test that everything works
        np.testing.assert_allclose(an_sol.delta_Bc_over_Bc(a).real, sci_sol.delta_Bc_over_Bc(a).real, atol=1e-6)
        np.testing.assert_allclose(an_sol.delta_Bc_over_Bc(a).imag, sci_sol.delta_Bc_over_Bc(a).imag, atol=1e-6)
        np.testing.assert_allclose(an_sol.delta_u_over_va(a).real,  sci_sol.delta_u_over_va(a).real, atol=1e-6)
        np.testing.assert_allclose(an_sol.delta_u_over_va(a).imag,  sci_sol.delta_u_over_va(a).imag, atol=1e-6)

    if show:
        plt.show()


if __name__ == '__main__':
    test_alfven_wave_standing(show=True)
