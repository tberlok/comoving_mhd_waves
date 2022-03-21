import numpy as np


class AnalyticComovingAlfvenWave:
    def __init__(self, OmegaA, ai, A_u, A_B):
        self.OmegaA = OmegaA
        self.ai = ai
        self.A_u = A_u
        self.A_B = A_B

        self.kappa = np.sqrt(self.OmegaA**2 - 1/16, dtype=np.complex)

    def delta_Bc_over_Bc(self, a):
        kappa = self.kappa
        psi = kappa*np.log(a/self.ai)
        fac1 = self.A_B
        fac2 = (self.A_B +
                4*1j*self.OmegaA*np.sqrt(self.ai)*self.A_u)/(4*kappa)
        res = (a/self.ai)**(-1/4)*(fac1*np.cos(psi) + fac2*np.sin(psi))

        return res

    def delta_u_over_va(self, a):
        kappa = self.kappa
        psi = kappa*np.log(a/self.ai)
        fac1 = self.A_u
        fac2 = - (self.A_u -
                  4*1j*self.OmegaA/np.sqrt(self.ai)*self.A_B)/(4*kappa)
        res = (a/self.ai)**(-3/4)*(fac1*np.cos(psi) + fac2*np.sin(psi))

        return res
