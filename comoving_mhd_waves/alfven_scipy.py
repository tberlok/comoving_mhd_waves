class ScipyComovingAlfvenWave:

    def __init__(self, OmegaA, ai, A_u, A_B, Omega0=1, OmegaLambda=0,
                 points=10000):
        import numpy as np
        from scipy.integrate import solve_ivp

        def rhs(a, y):
            """
            RHS
            """
            f = [0, 0]
            dBc_over_Bc = y[0]
            a_times_du_over_va = y[1]

            adot_over_H0 = a*np.sqrt(Omega0/a**3 +
                                     (1 - Omega0 - OmegaLambda)/a**2
                                     + OmegaLambda)

            f[0] = 1j*OmegaA/(a**2*adot_over_H0) * a_times_du_over_va
            f[1] = 1j*OmegaA/(a*adot_over_H0) * dBc_over_Bc

            return f

        f0 = np.array([A_B, ai*A_u], dtype=np.complex)

        self.sol = solve_ivp(rhs, [ai, 1], f0, method='BDF',
                             dense_output=True, atol=1e-10, rtol=1e-8)

        self.OmegaA = OmegaA
        self.ai = ai
        self.A_u = A_u
        self.A_B = A_B
        self.Omega0 = Omega0
        self.OmegaLambda = OmegaLambda
        self.points = points

    def delta_Bc_over_Bc(self, a):
        y = self.sol.sol(a)
        dBc_over_Bc = y[0]
        return dBc_over_Bc

    def delta_u_over_va(self, a):
        y = self.sol.sol(a)
        du_over_va = y[1]/a
        return du_over_va
