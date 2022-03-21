import numpy as np


class ScipyComovingMagnetosonicWave:

    def __init__(self, k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho,
                 Omega0=1, OmegaLambda=0, points=10000):
        from scipy.integrate import solve_ivp

        def rhs(a, y):
            """
            RHS
            """
            f = [0, 0]
            drhoc_over_rhoc = y[0]
            a_times_du = y[1]

            adot = a*H0*np.sqrt(Omega0/a**3 + (1 - Omega0 - OmegaLambda)/a**2
                                + OmegaLambda)

            f[0] = -1j*k/(a**2*adot) * a_times_du
            f[1] = -1j*k/adot*(Vs**2/a**(3*(gamma-1))
                               + (Va**2 - Vg**2)/a) * drhoc_over_rhoc

            return f

        f0 = np.array([A_rho, ai*A_u*Vs], dtype=np.complex)

        sol = solve_ivp(rhs, [ai, 1], f0, method='BDF',
                        dense_output=True, atol=1e-10, rtol=1e-8)

        self.k = k
        self.H0 = H0
        self.Vs = Vs
        self.Va = Va
        self.Vg = Vg
        self.gamma = gamma
        self.ai = ai
        self.A_u = A_u
        self.A_rho = A_rho
        self.Omega0 = Omega0
        self.OmegaLambda = OmegaLambda
        self.points = points

        self.sol = sol

        self.OmegaS = k*Vs/H0
        self.OmegaA = k*Va/H0
        self.OmegaG = k*Vg/H0

    def delta_rhoc_over_rhoc(self, a):
        y = self.sol.sol(a)
        return y[0]

    def delta_u_over_vs(self, a):
        y = self.sol.sol(a)
        return y[1]/a/self.Vs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from compressible_class import ComovingMagnetosonicWave

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

    an_sol = ComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)
    sci_sol = ScipyComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)

    a = np.logspace(np.log10(ai), 0, 2000)
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
    plt.show()
