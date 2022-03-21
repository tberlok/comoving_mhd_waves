import numpy as np
from .bessel_functions import J, J_p, Y, Y_p


class AnalyticComovingMagnetosonicWave:

    def __init__(self, k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho):
        # TODO: Add gamma = 4/3 solution as well or put it somewhere else.
        msg = 'Code does not work for gamma=4/3. It does work for e.g. 4/3+1e-6'
        assert gamma != 4/3, msg

        import numpy as np
        self.A_u = A_u
        self.A_rho = A_rho
        self.ai = ai
        self.s = (4 - 3*gamma)/2
        self.gamma = gamma
        self.zeta = k*Va/H0
        self.xi = k*Vs/H0
        zeta_g = k*Vg/H0
        self.nu = np.sqrt(1 - 16*(self.zeta**2 - zeta_g**2), dtype=np.complex)/(4*np.abs(self.s))

        # self.d =  self.h1(ai)*self.h2p(ai) - self.h2(ai)*self.h1p(ai)
        self.c1 = self.A_rho*self.h2p(ai) + 1j*self.A_u*self.xi/np.sqrt(ai)*self.h2(ai)
        self.c1 *= np.pi*ai**(3/2)/(2*self.s)

        self.c2 = self.A_rho*self.h1p(ai) + 1j*self.A_u*self.xi/np.sqrt(ai)*self.h1(ai)
        self.c2 *= -np.pi*ai**(3/2)/(2*self.s)

    def h1(self, a):
        z = self.xi*a**self.s/np.abs(self.s)
        return a**(-1/4)*J(self.nu, z)

    def h2(self, a):
        z = self.xi*a**self.s/np.abs(self.s)
        return a**(-1/4)*Y(self.nu, z)

    def h1p(self, a):
        z = self.xi*a**self.s/np.abs(self.s)
        sgn_s = np.sign(self.s)
        y = -1/4*J(self.nu, z)
        y += self.xi*sgn_s * a**self.s * J_p(self.nu, z)
        return y*a**(-5/4)

    def h2p(self, a):
        z = self.xi*a**self.s/np.abs(self.s)
        sgn_s = np.sign(self.s)
        y = -1/4*Y(self.nu, z)
        y += self.xi*sgn_s * a**self.s * Y_p(self.nu, z)
        return y*a**(-5/4)

    # def delta_rhoc_over_rhoc(self, a):
    #     z = self.xi*a**self.s/np.abs(self.s)
    #     return a**(-1/4)*(self.c1*J(self.nu, z) + self.c2*Y(self.nu, z))

    # def delta_u_over_vs(self, a):
    #     """ This is the equation in the paper, the one below is
    #         somehow nicer """
    #     z = self.xi*a**self.s/np.abs(self.s)
    #     xi = self.xi
    #     sgn_s = np.sign(self.s)
    #     s = self.s
    #     res = -1j/(4*xi*np.sqrt(a))*self.delta_rhoc_over_rhoc(a)
    #     res += 1j*sgn_s*a**(s-3/4)*(self.c1*J_p(self.nu, z) +
    #                                 self.c2*Y_p(self.nu, z))
    #     return res

    def delta_rhoc_over_rhoc(self, a):
        return self.c1*self.h1(a) + self.c2*self.h2(a)

    def delta_u_over_vs(self, a):
        drho_da = self.c1*self.h1p(a) + self.c2*self.h2p(a)
        return 1j*np.sqrt(a)/self.xi*drho_da


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ai = 1/128
    A_u = 1
    A_rho = 0

    # gamma = 4/3 + 1e-6
    gamma = 5/3
    rhoc = 1
    Bc = 0.
    H0 = 1
    k = 2*np.pi
    pc_0 = 3/5/100
    beta0 = 8*pc_0
    Bc = np.sqrt(2*pc_0/beta0)
    a = np.linspace(ai, 1, 2000)
    a = np.logspace(np.log10(ai), 0, 300)
    plt.figure(1)
    plt.clf()
    fig, axes = plt.subplots(num=1, nrows=2, ncols=2, sharex=True)

    for G in [0, 0.2]:

        # 'Velocities'
        Vs = np.sqrt(gamma*pc_0/rhoc)
        Va = Bc/np.sqrt(rhoc)
        Vg = np.sqrt(4*np.pi*G*rhoc)/k

        an_sol = AnalyticComovingMagnetosonicWave(k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho)

        axes[0, 0].semilogx(a, an_sol.delta_rhoc_over_rhoc(a).real, label='Analytic')
        axes[0, 1].semilogx(a, an_sol.delta_rhoc_over_rhoc(a).imag)
        axes[1, 0].semilogx(a, an_sol.delta_u_over_vs(a).real)
        axes[1, 1].semilogx(a, an_sol.delta_u_over_vs(a).imag)
        axes[0, 0].set_title(r'$\mathrm{Re}(\delta \rho_\mathrm{c}/\rho_\mathrm{c})$')
        axes[0, 1].set_title(r'$\mathrm{Im}(\delta \rho_\mathrm{c}/\rho_\mathrm{c})$')
        axes[1, 0].set_title(r'$\mathrm{Re}(\delta u/\mathcal{V}_\mathrm{A})$')
        axes[1, 1].set_title(r'$\mathrm{Im}(\delta u/\mathcal{V}_\mathrm{A})$')

    plt.show()
