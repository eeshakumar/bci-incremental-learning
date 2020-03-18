import numpy as np
import pywt


def dwt_features(data, trials, level, sampling_freq, w, n, wavelet, n_features=12):

    x = np.zeros((len(trials), n_features))
    w_lo = w[0]
    w_hi = w[1]

    for t, trial in enumerate(trials):
        signals = data[trial + sampling_freq * 4 + w_lo:
                       trial + sampling_freq * 4 + w_hi]
        dwt_c3 = discrete_transform(signals, 0, wavelet, level)
        dwt_cz = discrete_transform(signals, 1, wavelet, level)
        dwt_c4 = discrete_transform(signals, 2, wavelet, level)

        x[t,:] = generate_samples(n, dwt_c3, dwt_cz, dwt_c4)
    return x


def discrete_transform(signals, idx, wavelet, level):
    return pywt.wavedec(signals[:, idx], wavelet=wavelet, level=level)


def generate_samples(n, dwt_c3, dwt_cz, dwt_c4):

    return np.array([
        np.std(dwt_c3[n]),
        np.std(dwt_cz[n]),
        np.std(dwt_c4[n]),
        np.sqrt(np.mean(np.square(dwt_c3[n]))),
        np.sqrt(np.mean(np.square(dwt_cz[n]))),
        np.sqrt(np.mean(np.square(dwt_c4[n]))),
        np.mean(dwt_c3[n] ** 2),
        np.mean(dwt_cz[n] ** 2),
        np.mean(dwt_c4[n] ** 2),
        np.mean(dwt_c3[n]),
        np.mean(dwt_cz[n]),
        np.mean(dwt_c4[n])])
