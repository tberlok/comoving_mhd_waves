import numpy as np


def drhoc_over_rhoc(x, a, direction='right'):
    if direction == 'right':
        sign = -1
    elif direction == 'left':
        sign = 1
    else:
        raise RuntimeError('direction should be left or right')
    psi = kappa*np.log(a/ai)
    phase = k*x + sign*psi
    f = -np.sqrt(ai)*OmegaS/(4*sigma**2) * (sign*4*kappa * np.cos(phase)
                                            + np.sin(phase))*(a/ai)**(-1/4)
    return f.real


def du_over_vs(x, a, direction='right'):
    if direction == 'right':
        sign = -1
    elif direction == 'left':
        sign = 1
    else:
        raise RuntimeError('direction should be left or right')
    psi = kappa*np.log(a/ai)
    phase = k*x + sign*psi
    f = np.cos(phase)*(a/ai)**(-3/4)
    return f.real


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Animation of traveling Alvén wave.
    # Parameters as in Fig. 7 in the paper

    direc = 'right'

    mhd = True
    selfgravity = False

    k = 2*np.pi
    H0 = 1
    Vs = 1/2
    Va = 1/2
    Vg = 1/4

    if not mhd:
        Va = 0
    if not selfgravity:
        Vg = 0

    OmegaS = k*Vs/H0
    OmegaA = k*Va/H0
    OmegaG = k*Vg/H0

    sigma = np.sqrt(OmegaS**2 + OmegaA**2 - OmegaG**2, dtype=np.complex)
    kappa = np.sqrt(sigma**2 - 1/16, dtype=np.complex)

    n_periods = 3
    ai = np.exp(-2*n_periods*np.pi/kappa).real

    x = np.linspace(0, 1, 256)

    plt.figure(2)
    plt.clf()

    fig, axes = plt.subplots(num=2, nrows=2, ncols=2, sharex=True,
                             sharey='row')

    line1, = axes[0, 0].plot(x, drhoc_over_rhoc(x, ai, direc))
    line2, = axes[0, 1].plot(x, drhoc_over_rhoc(x, ai, direc))
    line3, = axes[1, 0].plot(x, du_over_vs(x, ai, direc))
    line4, = axes[1, 1].plot(x, du_over_vs(x, ai, direc))
    axes[0, 0].plot(x, drhoc_over_rhoc(x, ai, direc), 'k:')
    axes[0, 1].plot(x, drhoc_over_rhoc(x, ai, direc), 'k:')
    axes[1, 0].plot(x, du_over_vs(x, ai, direc), 'k:')
    axes[1, 1].plot(x, du_over_vs(x, ai, direc), 'k:')

    for ii in range(2):
        axes[1, ii].set_xlabel(r'$x/L$')

    axes[0, 0].set_ylabel(r'$\delta \rho_\mathrm{c}/\rho_\mathrm{c}$')
    axes[0, 1].set_ylabel(r'$(a/a_\mathrm{i})^{1/4}\delta \rho_\mathrm{c}/\rho_\mathrm{c}$')
    axes[1, 0].set_ylabel(r'$\delta u/\mathcal{V}_\mathrm{s}$')
    axes[1, 1].set_ylabel(r'$(a/a_\mathrm{i})^{3/4}\delta u/\mathcal{V}_\mathrm{s}$')

    for a in np.logspace(np.log10(ai), 0, 200):

        line1.set_ydata(drhoc_over_rhoc(x, a, direc))
        line2.set_ydata((a/ai)**(1/4)*drhoc_over_rhoc(x, a, direc))
        line3.set_ydata(du_over_vs(x, a, direc))
        line4.set_ydata((a/ai)**(3/4)*du_over_vs(x, a, direc))
        fig.suptitle(r'$\ln(a/ai) = ' + '{:1.2f}'.format(np.log(a/ai)) + '$')
        plt.pause(1e-2)

    plt.show()
