import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from commpy import filters
from commpy import impairments
from rtlsdr import RtlSdr

samples_per_symbol = 8
num_samples = 65536 * 8

sdr = RtlSdr()

# configure device
sdr.sample_rate = 1e6  # Hz
sdr.center_freq = 920e6     # Hz
sdr.gain = 'auto'

rx = sdr.read_samples(num_samples)
rx = sdr.read_samples(num_samples)

# Generate root raised cosine
num_taps = 50 * samples_per_symbol + int (samples_per_symbol / 2)
alpha = 0.35
t, h = filters.rrcosfilter(num_taps, alpha, 1, samples_per_symbol)

#FFT of signal
s = rx * np.hamming(len(rx))
S = np.fft.fftshift(np.fft.fft(s))
S_mag = np.abs(S)
f = np.linspace(-0.5,0.5,len(rx))
plt.figure(0)
plt.plot(f, S_mag,'.-')
plt.show()

#Receive

# Coarse carrier recovery
psd = np.abs(np.fft.fft(rx ** 4))
f = np.linspace(-1/2.0, 1/2.0, len(psd))
plt.plot(f, psd)
plt.show()

print("Carrier offset is: ")
print(np.argmax(psd) / len(rx) / 4)

rx = impairments.add_frequency_offset(rx, len(rx), -np.argmax(psd) / 4)
psd = np.abs(np.fft.fft(rx ** 4))
print("Carrier offset (post coarse correction) is: ")
print(np.argmax(psd))

# Matched filter
matched_filtered = np.convolve(rx, h)

#Symbol timing recovery

#Upsample
timing_upsample = 16
interpolated = signal.resample_poly(matched_filtered, timing_upsample, 1)
samples_per_symbol_interp = timing_upsample * samples_per_symbol

i_in = 0;
i_out = 0;
timing_out = np.zeros(int(num_samples / samples_per_symbol), dtype=complex)
while i_out < len(timing_out) and i_in < len(interpolated):
    i_in_int = int(round(i_in))

    timing_out[i_out] = interpolated[i_in_int]

    timing_error = ((interpolated[i_in_int] - interpolated[i_in_int - samples_per_symbol_interp]) * interpolated[i_in_int - int(samples_per_symbol_interp/2)].conj()).real;

    i_out += 1
    i_in += samples_per_symbol_interp - timing_error * 0.005

#Plot symbol timing convergence
plt.figure(0)
plt.plot(abs(timing_out), '.')
plt.show()

#Fine frequency recovery
phase = 0
freq = 0

alpha = 0.01
beta = 0.002

out = np.zeros(len(timing_out), dtype=np.complex)

for i in range(len(timing_out)):
    out[i] = timing_out[i] * np.exp(-1j*phase) # adjust the input sample by the inverse of the estimated phase offset
    error = np.sign(np.real(out[i])) * np.imag(out[i]) - np.sign(np.imag(out[i])) * np.real(out[i])

    # Advance the loop (recalc phase and freq offset)
    freq += (beta * error)
    phase += freq + (alpha * error)

    # Adjust phase so its always between 0 and 2pi
    while phase >= 2*np.pi:
        phase -= 2*np.pi
    while phase < 0:
        phase += 2*np.pi

#Plot carrier phase
plt.figure(0)
plt.plot(out.real, '.')
plt.show()

#Slicer
out_flat = out.view(np.float64)
out_sliced = out_flat > 0
#Descrambler
out_descrambled = out_sliced[:-23] ^ out_sliced[5:-18] ^ out_sliced[23:]
print(out_descrambled[15000:15100])

