import numpy as np

def ezfft(t_, f_t_):
    # function [omega, F] = ezfft(t, f)
    # zhyang
    # Compute the forward fft of a function f(t) that is defined in a symmetric range:
    # (1) t is increasing from -t_max to +t_max
    # (2) the length of t is odd
    # (3) t is equally spaced
    # f(t) - > F(omega) = \int_{-\infty}^\infty dt exp(-i \omega t) f(t)
    
    delta_t = t_[1]-t_[0]
    t_max = t_[-1]
    N = (len(t_) + 1) / 2
    
    tt = 2 * t_max * (2*N-1) / (2*(N-1))
    delta_omega = 2 * np.pi / tt
    omega_ = np.arange(-(N-1), (N-1)+1)*delta_omega

    f_t_shift = np.fft.ifftshift(f_t_)
    f_omega_shift = np.fft.fft(f_t_shift) * delta_t
    f_omega_ = np.fft.fftshift(f_omega_shift)
    return omega_, f_omega_

def ezifft(omega_, f_omega_):
    # function [t, f] = ezifft(omega, F)
    # zhyang
    # Compute the inverse fft of a function F(omega) that is defined in a symmetric range:
    # (1) omega is increasing from -omega_max to +omega_max
    # (2) the length of omega is odd
    # (3) omega is equally spaced
    # F(omega) - > f(t) = 1/(2*pi) \int_{-\infty}^\infty d\omega exp(+i \omega t) F(\omega)

    delta_omega = omega_[1]-omega_[0]
    omega_max = omega_[-1]
    N = (len(omega_) + 1) / 2

    oo = 2 * omega_max * (2*N-1) / (2*(N-1))
    delta_t = 2 * np.pi / oo
    t_ = np.arange(-(N-1), (N-1)+1) * delta_t

    f_omega_shift = np.fft.ifftshift(f_omega_)
    f_t_shift = np.fft.ifft(f_omega_shift) / delta_t; # delta_t * delta_omega = np.pi/(N-1)
    f_t_ = np.fft.fftshift(f_t_shift)
    return t_, f_t_