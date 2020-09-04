import numpy as np

def gen_mix(s, n, snr, energy_norm=True):
    if (np.sum(n**2) * (10**(snr / 10))) == 0:
        alpha = 1
    else:
        alpha = np.sqrt(np.sum(s**2.0) / (np.sum(n**2.0) * (10.0**(snr / 10.0))))

    alpha = np.where(np.isnan(alpha), np.zeros_like(alpha), alpha)
    m = s + n * alpha

    if energy_norm:
        CONSTANT = 0.95  #energy nomalization
        c = np.sqrt(CONSTANT * m.size / np.sum(m**2))
    else:
        peak = np.max(np.abs(m))
        if peak > 4:
            m = np.clip(m, -4, 4)
            c = 0.25
        elif peak > 1:
            c = 1 / peak
        else:
            c = 1

    return s * c, n * alpha * c, m * c

