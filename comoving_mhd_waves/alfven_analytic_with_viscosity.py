import numpy as np


class AnalyticComovingAlfvenWaveWithViscosity:
    """
    Implementation of equations C26 & C27 in the paper.
    For Ωa = (1 + 2*Gamma)/4, the solution is slightly different.
    """
    def __init__(self, OmegaA, ai, A_u, A_B, Gamma=0):
        self.OmegaA = OmegaA
        self.Gamma = Gamma
        self.ai = ai
        self.A_u = A_u
        self.A_B = A_B

        self.kappa = np.sqrt(self.OmegaA**2 - (1+2*Gamma)**2/16,
                             dtype=np.complex128)

    def delta_Bc_over_Bc(self, a):
        if np.isclose(self.OmegaA, (1 + 2*self.Gamma)/4):
            fac1 = self.A_B
            fac2 = (1+2*self.Gamma)*(self.A_B + 1j*np.sqrt(self.ai)*self.A_u)/4
            res = (a/self.ai)**(-1/4)*(fac1 + fac2*np.log(a/self.ai))
        else:
            kappa = self.kappa
            psi = kappa*np.log(a/self.ai)
            fac1 = self.A_B
            fac2 = (self.A_B*(1+2*self.Gamma) +
                    4*1j*self.OmegaA*np.sqrt(self.ai)*self.A_u)/(4*kappa)
            res = (a/self.ai)**(-1/4)*(fac1*np.cos(psi) + fac2*np.sin(psi))

        return res*(a/self.ai)**(-self.Gamma/2)

    def delta_u_over_va(self, a):
        if np.isclose(self.OmegaA, (1 + 2*self.Gamma)/4):
            fac1 = self.A_u
            fac2 = (1+2*self.Gamma)*(-self.A_u + 1j/np.sqrt(self.ai)*self.A_B)/4
            res = (a/self.ai)**(-3/4)*(fac1 + fac2*np.log(a/self.ai))
        else:
            kappa = self.kappa
            psi = kappa*np.log(a/self.ai)
            fac1 = self.A_u
            fac2 = - (self.A_u*(1+2*self.Gamma) -
                      4*1j*self.OmegaA/np.sqrt(self.ai)*self.A_B)/(4*kappa)
            res = (a/self.ai)**(-3/4)*(fac1*np.cos(psi) + fac2*np.sin(psi))

        return res*(a/self.ai)**(-self.Gamma/2)
