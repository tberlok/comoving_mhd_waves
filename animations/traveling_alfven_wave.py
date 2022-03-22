import numpy as np


def dBc_over_Bc(x, a, direction='right'):
    """
    Traveling Alfvén wave solution
    """
    if direction == 'right':
        sign = -1
    elif direction == 'left':
        sign = 1
    else:
        raise RuntimeError('direction should be left or right')
    psi = sign*kappa*np.log(a/ai)
    phase = k*x + psi
    res = np.sqrt(ai)/(4*OmegaA) * (sign*4*kappa * np.cos(phase)
                                    + np.sin(phase))*(a/ai)**(-1/4)
    return res


def du_over_va(x, a, direction='right'):
    """
    Traveling Alfvén wave solution
    """
    if direction == 'right':
        sign = -1
    elif direction == 'left':
        sign = 1
    else:
        raise RuntimeError('direction should be left or right')
    psi = sign*kappa*np.log(a/ai)
    phase = k*x + psi
    return np.cos(phase)*(a/ai)**(-3/4)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Animation of traveling Alvén wave.
    # Parameters as in Fig. 7 in the paper

    direc = 'right'

    k = 2*np.pi
    H0 = 1
    Va = 1/2
    OmegaA = k*Va/H0

    kappa = np.sqrt(OmegaA**2 - 1/16)

    n_periods = 2
    ai = np.exp(-2*n_periods*np.pi/kappa)

    x = np.linspace(0, 1, 256)

    plt.figure(2)
    plt.clf()
    fig, axes = plt.subplots(num=2, nrows=2, ncols=2, sharex=True,
                             sharey='row')

    line1, = axes[0, 0].plot(x, dBc_over_Bc(x, ai, direc))
    line2, = axes[0, 1].plot(x, dBc_over_Bc(x, ai, direc))
    line3, = axes[1, 0].plot(x, du_over_va(x, ai, direc))
    line4, = axes[1, 1].plot(x, du_over_va(x, ai, direc))
    axes[0, 0].plot(x, dBc_over_Bc(x, ai, direc), 'k:')
    axes[0, 1].plot(x, dBc_over_Bc(x, ai, direc), 'k:')
    axes[1, 0].plot(x, du_over_va(x, ai, direc), 'k:')
    axes[1, 1].plot(x, du_over_va(x, ai, direc), 'k:')

    for ii in range(2):
        axes[1, ii].set_xlabel(r'$a$')

    axes[0, 0].set_ylabel(r'$\delta B_\mathrm{c}/B_\mathrm{c}$')
    axes[0, 1].set_ylabel(r'$(a/a_\mathrm{i})^{1/4}\delta B_\mathrm{c}/B_\mathrm{c}$')
    axes[1, 0].set_ylabel(r'$\delta u/\mathcal{V}_\mathrm{A}$')
    axes[1, 1].set_ylabel(r'$(a/a_\mathrm{i})^{3/4}\delta u/\mathcal{V}_\mathrm{A}$')

    for a in np.logspace(np.log10(ai), 0, 200):

        line1.set_ydata(dBc_over_Bc(x, a, direc))
        line2.set_ydata((a/ai)**(1/4)*dBc_over_Bc(x, a, direc))
        line3.set_ydata(du_over_va(x, a, direc))
        line4.set_ydata((a/ai)**(3/4)*du_over_va(x, a, direc))
        fig.suptitle(r'$\ln(a/ai) = ' + '{:1.2f}'.format(np.log(a/ai)) + '$')
        plt.pause(1e-2)

    plt.show()
