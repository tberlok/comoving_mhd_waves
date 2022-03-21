import numpy as np
from .bessel_functions import J, J_p, Y, Y_p


class AnalyticComovingMagnetosonicWave:
    """
    This class contains all the analytic solutions derived in the paper.
    The solutions are divided into those for γ=4/3 and those with γ≠4/3.
    In addition, the solution for γ=4/3 with σ=1/4 is given special treatment.
    """

    def __init__(self, k, H0, Vs, Va, Vg, gamma, ai, A_u, A_rho):

        self.A_u = A_u
        self.A_rho = A_rho
        self.ai = ai
        self.gamma = gamma
        self.OmegaA = k*Va/H0
        self.OmegaS = k*Vs/H0
        self.OmegaG = k*Vg/H0

        if gamma == 4/3:
            self.sigma = np.sqrt(self.OmegaS**2 + self.OmegaA**2
                                 - self.OmegaG**2, dtype=np.complex)
            self.kappa = np.sqrt(self.sigma**2 - 1/16, dtype=np.complex)
        else:
            self.s = (4 - 3*gamma)/2
            self.nu = np.sqrt(1 - 16*(self.OmegaA**2 - self.OmegaG**2),
                              dtype=np.complex)/(4*np.abs(self.s))

            self.c1 = self.A_rho*self.Gp(ai) + \
                1j*self.A_u*self.OmegaS/np.sqrt(ai)*self.G(ai)
            self.c1 *= np.pi*ai**(3/2)/(2*self.s)

            self.c2 = self.A_rho*self.Fp(ai) + \
                1j*self.A_u*self.OmegaS/np.sqrt(ai)*self.F(ai)
            self.c2 *= -np.pi*ai**(3/2)/(2*self.s)

    def F(self, a):
        """
        This is mathcal{F} as defined in the paper.
        """
        z = self.OmegaS*a**self.s/np.abs(self.s)
        return a**(-1/4)*J(self.nu, z)

    def G(self, a):
        """
        This is mathcal{G} as defined in the paper.
        """
        z = self.OmegaS*a**self.s/np.abs(self.s)
        return a**(-1/4)*Y(self.nu, z)

    def Fp(self, a):
        """
        The derivative of F
        """
        z = self.OmegaS*a**self.s/np.abs(self.s)
        sgn_s = np.sign(self.s)
        y = -1/4*J(self.nu, z)
        y += self.OmegaS*sgn_s * a**self.s * J_p(self.nu, z)
        return y*a**(-5/4)

    def Gp(self, a):
        """
        The derivative of G
        """
        z = self.OmegaS*a**self.s/np.abs(self.s)
        sgn_s = np.sign(self.s)
        y = -1/4*Y(self.nu, z)
        y += self.OmegaS*sgn_s * a**self.s * Y_p(self.nu, z)
        return y*a**(-5/4)

    def delta_rhoc_over_rhoc(self, a):
        if self.gamma == 4/3:
            if self.kappa == 0:
                fac1 = self.A_rho
                fac2 = (self.A_rho - 4*1j*self.OmegaS*np.sqrt(self.ai)*self.A_u)/4*np.log(a/self.ai)
                res = (a/self.ai)**(-1/4)*(fac1 + fac2)
            else:
                psi = self.kappa*np.log(a/self.ai)
                fac1 = self.A_rho*np.cos(psi)
                fac2 = (self.A_rho - 4*1j*self.OmegaS*np.sqrt(self.ai)*self.A_u)/(4*self.kappa)*np.sin(psi)
                res = (a/self.ai)**(-1/4)*(fac1 + fac2)
        else:
            res = self.c1*self.F(a) + self.c2*self.G(a)
        return res.astype(complex)

    def delta_u_over_vs(self, a):
        if self.gamma == 4/3:
            if self.kappa == 0:
                fac1 = self.A_u
                fac2 = -(1j*self.A_rho/np.sqrt(self.ai) + 4*self.A_u*self.OmegaS)/(16*self.OmegaS)*np.log(a/self.ai)
                res = (a/self.ai)**(-3/4)*(fac1 + fac2)
            else:
                psi = self.kappa*np.log(a/self.ai)
                fac1 = self.A_u*np.cos(psi)
                fac2 = -(self.A_u/(4*self.kappa) + 1j*self.A_rho*(1 + 16*self.kappa**2)/(16*np.sqrt(self.ai)*self.kappa*self.OmegaS)
                         )*np.sin(psi)
                res = (a/self.ai)**(-3/4)*(fac1 + fac2)
        else:
            drho_da = self.c1*self.Fp(a) + self.c2*self.Gp(a)
            res = 1j*np.sqrt(a)/self.OmegaS*drho_da
        return res.astype(complex)
