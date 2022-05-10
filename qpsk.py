import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from commpy import filters
from commpy import impairments

num_symbols = 32768
samples_per_symbol = 8

# Generate easily recognisable data consisting of alternating ones and zeros
bits = np.tile([1, 0, 1, 0, 1, 0, 1, 0], int((num_symbols * 2) / 8)) 

#Synchronous scrambler to make the data look random
scrambled = np.zeros(len(bits), dtype=bool)
for idx in range(len(bits)):
    scrambled[idx] = bits[idx] ^ scrambled[idx - 39] ^ scrambled[idx - 58]

# Upsample
x = np.zeros(num_symbols * samples_per_symbol, dtype=complex)
for idx, bitss in enumerate(scrambled.reshape((num_symbols, 2))):
    x[idx * samples_per_symbol] = (bitss[0]*2-1) + 1j*(bitss[1]*2-1)

# Generate root raised cosine filter 
num_taps = 20 * samples_per_symbol 
alpha = 0.35
t, rrc_taps = filters.rrcosfilter(num_taps, alpha, 1, samples_per_symbol)

#plt.figure(0)
#plt.plot(rrc_taps,'.-')
#plt.show()

# Interpolate with RRC
tx = np.convolve(x, rrc_taps)

#FFT of tx signal
#s = tx * np.hamming(len(tx))
#S = np.fft.fftshift(np.fft.fft(s))
#S_mag = np.abs(S)
#f = np.linspace(-0.5,0.5,len(tx))
#plt.figure(0)
#plt.plot(f, S_mag,'.-')
#plt.show()

#Channel
rx = tx
#Add delay
pre = np.zeros(99, dtype=complex)
rx = np.concatenate((pre, rx))
#Add noise
n = (np.random.randn(len(rx)) + 1j*np.random.randn(len(rx)))/np.sqrt(2)
rx = rx + 0.1 * n
#Add frequency offset
rx = impairments.add_frequency_offset(rx, 1, 0.2)

#Receive

# Coarse carrier recovery
psd = np.abs(np.fft.fft(rx ** 4) * np.hamming(len(rx)))
#f = np.linspace(-1/2.0, 1/2.0, len(psd))
#plt.plot(f, psd)
#plt.show()

print("Carrier offset is: ")
print(np.argmax(psd) / len(rx) / 4)

rx = impairments.add_frequency_offset(rx, len(rx), -np.argmax(psd) / 4)
psd = np.abs(np.fft.fft(rx ** 4))
print("Carrier offset (post coarse correction) is: ")
print(np.argmax(psd))

# Matched filter
matched_filtered = np.convolve(rx, rrc_taps) / samples_per_symbol

#Symbol timing recovery
#Upsample
timing_upsample = 8
interpolated = signal.resample_poly(matched_filtered, timing_upsample, 1)
samples_per_symbol_interp = timing_upsample * samples_per_symbol

i_in = 0;
i_out = 0;
timing_out = np.zeros(num_symbols, dtype=complex)
while i_out < len(timing_out) and i_in < len(interpolated):
    i_in_int = int(round(i_in))

    timing_out[i_out] = interpolated[i_in_int]

    timing_error = ((interpolated[i_in_int] - interpolated[i_in_int - samples_per_symbol_interp]) * interpolated[i_in_int - int(samples_per_symbol_interp/2)].conj()).real;

    i_out += 1
    i_in += samples_per_symbol_interp - timing_error * 0.3

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

#Plot real part of symbols
plt.figure(0)
plt.plot(out.real, '.')
plt.show()

#Slicer out the bits
out_flat = out.view(np.float64)
out_sliced = out_flat > 0
#Descramble
out_descrambled = out_sliced[:-58] ^ out_sliced[19:-39] ^ out_sliced[58:]
print(out_descrambled[7000:7100])

