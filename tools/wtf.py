import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# =====================================================
# 参数
# =====================================================

dt = 0.001

file_q30  = "Q30.bin"
file_q100 = "Q100.bin"

nx = 601
cpml = 20

source_x = 300

time_window = (0.25, 0.6)

fmax = 100

# =====================================================
# 读取自激自收道
# =====================================================

def load_trace(filename):

    data = np.fromfile(
        filename,
        dtype=np.float32
    ).reshape((-1, nx))

    data = data[:, 3+cpml:nx-4-cpml]

    trace = (
        data[:, source_x-3-cpml]
        .astype(np.float64)
        * 1e7
    )

    return trace


# =====================================================
# FFT
# =====================================================

def compute_fft(trace):

    nt = len(trace)

    t = np.arange(nt) * dt

    it1 = int(time_window[0] / dt)
    it2 = int(time_window[1] / dt)

    trace_win = trace[it1:it2]

    freq = fftfreq(
        len(trace_win),
        dt
    )

    amp = np.abs(
        fft(trace_win)
    )

    mask = (
        (freq >= 0)
        &
        (freq <= fmax)
    )

    return freq[mask], amp[mask]


# =====================================================
# 主程序
# =====================================================

trace30  = load_trace(file_q30)
trace100 = load_trace(file_q100)

t = np.arange(len(trace30))*dt

# FFT
freq30, amp30 = compute_fft(trace30)
freq100, amp100 = compute_fft(trace100)

# 谱重心

fc30 = np.sum(freq30*amp30)/np.sum(amp30)
fc100 = np.sum(freq100*amp100)/np.sum(amp100)

print(f"Q=30  Spectral Centroid = {fc30:.3f} Hz")
print(f"Q=100 Spectral Centroid = {fc100:.3f} Hz")

# 谱比

ratio = (
    amp100 + 1e-12
) / (
    amp30 + 1e-12
)

# =====================================================
# 绘图
# =====================================================

fig, ax = plt.subplots(
    1,
    3,
    figsize=(18,5)
)

# =====================================================
# 时域
# =====================================================

ax[0].plot(
    t,
    trace30,
    lw=1.5,
    label='Q=30'
)

ax[0].plot(
    t,
    trace100,
    lw=1.5,
    label='Q=100'
)

ax[0].set_xlim(time_window)

ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Amplitude")
ax[0].set_title("Time Domain Trace")

ax[0].grid(True)
ax[0].legend()

# =====================================================
# 振幅谱
# =====================================================

ax[1].plot(
    freq30,
    amp30,
    lw=2,
    label='Q=30'
)

ax[1].plot(
    freq100,
    amp100,
    lw=2,
    label='Q=100'
)

ax[1].set_xlim(0,fmax)

ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_ylabel("Amplitude")
ax[1].set_title("Amplitude Spectrum")

ax[1].grid(True)
ax[1].legend()

# =====================================================
# 谱比
# =====================================================

ax[2].plot(
    freq30,
    ratio,
    lw=2
)

ax[2].axhline(
    1.0,
    color='k',
    linestyle='--'
)

ax[2].set_xlim(0,fmax)

ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_ylabel("Q100 / Q30")
ax[2].set_title("Spectral Ratio")

ax[2].grid(True)

plt.tight_layout()
plt.show()