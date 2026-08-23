import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os

# ==========================================================
# 全局配置
# ==========================================================

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# 字体
FONT_LABEL = 12
FONT_TITLE = 14
FONT_TICK  = 10
FONT_LEGEND = 10

# 时间窗
TW_RMS = (0.15, 0.6)
TW_FFT = (0.15, 0.6)
TW_ST  = (0.15, 0.6)

# 最大绘图频率
F_MAX_PLOT = 100.0

DT = 0.001
FS = 1.0 / DT

# ==========================================================
# 第一组数据（Qf_30）
# ==========================================================

FILE_DIR_GROUP1 = [
    "./porosity/Qf_30/0.bin",
    "./porosity/Qf_30/5.bin",
    "./porosity/Qf_30/10.bin",
    "./porosity/Qf_30/15.bin",
    "./porosity/Qf_30/20.bin",
    "./porosity/Qf_30/25.bin",
]

LABELS_GROUP1 = ["0%", "5%", "10%", "15%", "20%", "25%"]

PARAMS_GROUP1 = [
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
]

POSX_GROUP1 = [300, 300, 300, 300, 300, 300]

# ==========================================================
# 第二组数据（Qf_9999）
# ==========================================================

FILE_DIR_GROUP2 = [
    "./porosity/Qf_9999/0.bin",
    "./porosity/Qf_9999/5.bin",
    "./porosity/Qf_9999/10.bin",
    "./porosity/Qf_9999/15.bin",
    "./porosity/Qf_9999/20.bin",
    "./porosity/Qf_9999/25.bin",
]

LABELS_GROUP2 = ["0%", "5%", "10%", "15%", "20%", "25%"]

PARAMS_GROUP2 = [
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
    {"nx": 601, "nz": 501, "dh": 1.5},
]

POSX_GROUP2 = [300, 300, 300, 300, 300, 300]

# ==========================================================
# 从路径中提取组名
# ==========================================================

def get_group_name_from_path(file_path):
    return os.path.basename(os.path.dirname(file_path))

GROUP1_NAME = get_group_name_from_path(FILE_DIR_GROUP1[0])
GROUP2_NAME = get_group_name_from_path(FILE_DIR_GROUP2[0])

# ==========================================================
# 频段划分
# ==========================================================

FREQ_BANDS = [
    (0, 40),
    (41, 100)
]

# ==========================================================
# 工具函数
# ==========================================================

def slice_by_window(trace_orig, t_full, window, dt):
    if window is not None:
        t_start, t_end = window
        idx_start = max(0, int(np.round(t_start / dt)))
        idx_end = min(len(trace_orig), int(np.round(t_end / dt)))
        return trace_orig[idx_start:idx_end].copy(), t_full[idx_start:idx_end].copy()
    return trace_orig.copy(), t_full.copy()


def calculate_rms_profile(data, dt, dx, time_window):
    if time_window is not None:
        t_start, t_end = time_window
        idx_start = max(0, int(np.round(t_start / dt)))
        idx_end = min(data.shape[0], int(np.round(t_end / dt)))
        win_data = data[idx_start:idx_end, :]
    else:
        win_data = data
    rms_values = np.sqrt(np.mean(np.square(win_data), axis=0))
    dist_x = (np.arange(len(rms_values)) - len(rms_values) // 2) * dx
    return dist_x, rms_values


def spectral_centroid(freq, amp):
    numerator = np.sum(freq * amp)
    denominator = np.sum(amp)
    if denominator <= 1e-14:
        return 0.0
    return numerator / denominator


def stran(x, t, f):
    x = x.ravel()
    t = t.ravel()
    f = f.ravel()

    Nt = len(t)
    Nf = len(f)
    dt = t[1] - t[0]

    STx = np.zeros((Nf, Nt), dtype=np.complex128)
    f_col = f[:, None]
    t_row = t[None, :]
    f_abs = np.abs(f_col)

    for i in range(Nt):
        tau = t[i]
        td = t_row - tau
        gauss = (f_abs / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (f_col ** 2) * (td ** 2))
        phase = np.exp(-2j * np.pi * f_col * t_row)
        STx[:, i] = np.sum(x * gauss * phase, axis=1) * dt
    return STx

# ==========================================================
# 主程序
# ==========================================================

def main():
    # ======================================================
    # 1. 处理第一组数据（Qf_30）
    # ======================================================
    num_files = len(FILE_DIR_GROUP1)
    data_records1 = []
    centroid_list1 = []
    st_results = []
    global_st_max = 0.0

    for i in range(num_files):
        nx_val = PARAMS_GROUP1[i]["nx"]
        dh_val = PARAMS_GROUP1[i]["dh"]
        cpml = 20

        temp = np.fromfile(FILE_DIR_GROUP1[i], dtype=np.float32).reshape((-1, nx_val))
        temp = temp[:, 3+cpml:nx_val-4-cpml]
        trace_orig = temp[:, POSX_GROUP1[i]-3-cpml].copy().astype(np.float64) * 1e7
        t_full = np.arange(len(trace_orig)) * DT

        trace_fft, t_fft = slice_by_window(trace_orig, t_full, TW_FFT, DT)
        freq_fft = fftfreq(len(trace_fft), DT)
        amp_fft = np.abs(fft(trace_fft))
        freq_pos = freq_fft[freq_fft >= 0]
        amp_pos = amp_fft[freq_fft >= 0]
        mask_plot = (freq_pos >= 0) & (freq_pos <= F_MAX_PLOT)
        freq_plot = freq_pos[mask_plot]
        amp_plot = amp_pos[mask_plot]

        centroid = spectral_centroid(freq_plot, amp_plot)
        centroid_list1.append(centroid)

        dist_x, rms_values = calculate_rms_profile(temp.astype(np.float64), DT, dh_val, TW_RMS)

        trace_st, t_st = slice_by_window(trace_orig, t_full, TW_ST, DT)
        f_s = np.linspace(0, FS / 2, len(trace_st) // 2)
        STx = stran(trace_st, t_st, f_s)
        s_amp = np.abs(STx)
        f_mask_s = (f_s <= F_MAX_PLOT)
        f_s_plot = f_s[f_mask_s]
        s_amp_plot = s_amp[f_mask_s, :]
        global_st_max = max(global_st_max, np.max(s_amp_plot))

        st_results.append({
            "t": t_st,
            "f": f_s_plot,
            "amp": s_amp_plot
        })

        data_records1.append({
            "label": LABELS_GROUP1[i],
            "t_fft": t_fft,
            "trace_fft": trace_fft,
            "freq_pos": freq_plot,
            "amp_pos": amp_plot,
            "dist_x": dist_x,
            "rms_values": rms_values
        })

    # ======================================================
    # 2. 处理第二组数据（Qf_9999），一次性提取所需信息
    # ======================================================
    group2_data = []
    centroid_list2 = []

    for i in range(num_files):
        nx_val = PARAMS_GROUP2[i]["nx"]
        cpml = 20

        temp = np.fromfile(FILE_DIR_GROUP2[i], dtype=np.float32).reshape((-1, nx_val))
        temp = temp[:, 3+cpml:nx_val-4-cpml]
        trace_orig = temp[:, POSX_GROUP2[i]-3-cpml].copy().astype(np.float64) * 1e7
        t_full = np.arange(len(trace_orig)) * DT

        trace_fft, t_fft = slice_by_window(trace_orig, t_full, TW_FFT, DT)
        freq_fft = fftfreq(len(trace_fft), DT)
        amp_fft = np.abs(fft(trace_fft))
        freq_pos = freq_fft[freq_fft >= 0]
        amp_pos = amp_fft[freq_fft >= 0]
        mask_plot = (freq_pos >= 0) & (freq_pos <= F_MAX_PLOT)
        freq_plot = freq_pos[mask_plot]
        amp_plot = amp_pos[mask_plot]

        centroid = spectral_centroid(freq_plot, amp_plot)
        centroid_list2.append(centroid)

        group2_data.append({
            "label": LABELS_GROUP2[i],
            "t_fft": t_fft,
            "trace_fft": trace_fft,
            "freq_pos": freq_plot,
            "amp_pos": amp_plot,
            "centroid": centroid
        })

    # ======================================================
    # 图1：Group1 时域波形 + 频谱
    # ======================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    for rec in data_records1:
        axes1[0].plot(rec['t_fft'], rec['trace_fft'], linewidth=1.5, label=rec['label'])
    axes1[0].set_title("Time Domain Waveforms", fontsize=FONT_TITLE)
    axes1[0].set_xlabel("Time (s)", fontsize=FONT_LABEL)
    axes1[0].set_ylabel("Amplitude", fontsize=FONT_LABEL)
    axes1[0].tick_params(labelsize=FONT_TICK)
    axes1[0].grid(True, linestyle='--', alpha=0.5)
    axes1[0].legend(fontsize=FONT_LEGEND)
    axes1[0].text(0.5, -0.18, '(a)', transform=axes1[0].transAxes, ha='center', va='top', fontsize=14)

    for rec in data_records1:
        axes1[1].plot(rec['freq_pos'], rec['amp_pos'], linewidth=1.5, label=rec['label'])
    axes1[1].set_title("Amplitude Spectra", fontsize=FONT_TITLE)
    axes1[1].set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    axes1[1].set_ylabel("Amplitude", fontsize=FONT_LABEL)
    axes1[1].set_xlim(0, F_MAX_PLOT)
    axes1[1].tick_params(labelsize=FONT_TICK)
    axes1[1].grid(True, linestyle='--', alpha=0.5)
    axes1[1].legend(fontsize=FONT_LEGEND)
    axes1[1].text(0.5, -0.18, '(b)', transform=axes1[1].transAxes, ha='center', va='top', fontsize=14)
    fig1.tight_layout()

    # ======================================================
    # 图2：Group1 地表 RMS 振幅
    # ======================================================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for rec in data_records1:
        ax2.plot(rec['dist_x'], rec['rms_values'], linewidth=2, label=rec['label'])
    ax2.set_title("Surface RMS Amplitude", fontsize=FONT_TITLE)
    ax2.set_xlabel("Offset (m)", fontsize=FONT_LABEL)
    ax2.set_ylabel("RMS Amplitude", fontsize=FONT_LABEL)
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=FONT_LEGEND)
    fig2.tight_layout()

    # ======================================================
    # 图3：Group1 S 变换
    # ======================================================
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes3 = axes3.flatten()
    for i in range(num_files):
        ax = axes3[i]
        st_data = st_results[i]
        im = ax.imshow(st_data["amp"], cmap='turbo', aspect='auto', origin='lower',
                       extent=[st_data["t"].min(), st_data["t"].max(),
                               st_data["f"].min(), st_data["f"].max()],
                       vmin=0, vmax=global_st_max)
        ax.set_title(f"$\phi$={LABELS_GROUP1[i]}", fontsize=FONT_TITLE)
        ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
        ax.set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
        ax.tick_params(labelsize=FONT_TICK, labelleft=True, labelbottom=True)
        ax.text(0.5, -0.18, f'({chr(97+i)})', transform=ax.transAxes, ha='center', va='top', fontsize=14)
        cbar = fig3.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=9)
    fig3.tight_layout()

    # ======================================================
    # 图4：谱重心对比 (Group1 vs Group2)
    # ======================================================
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(LABELS_GROUP1, centroid_list1, marker='o', linewidth=2, label=GROUP1_NAME)
    ax4.plot(LABELS_GROUP1, centroid_list2, marker='s', linewidth=2, label=GROUP2_NAME)
    ax4.set_title("Spectral Centroid", fontsize=FONT_TITLE)
    ax4.set_ylabel("Centroid Frequency (Hz)", fontsize=FONT_LABEL)
    ax4.set_xlabel("Attenuation level", fontsize=FONT_LABEL)
    ax4.tick_params(labelsize=FONT_TICK)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(fontsize=FONT_LEGEND)
    fig4.tight_layout()

    # ======================================================
    # 图5：Group1 频谱比
    # ======================================================
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ref_amp = data_records1[0]["amp_pos"]
    ref_freq = data_records1[0]["freq_pos"]
    for i in range(1, num_files):
        amp_pos = data_records1[i]["amp_pos"]
        ratio = (amp_pos + 1e-14) / (ref_amp + 1e-14)
        ax5.plot(ref_freq, ratio, linewidth=2, label=f"{LABELS_GROUP1[i]} / 0%")
    ax5.set_title("Linear Spectral Ratio (Group 1)", fontsize=FONT_TITLE)
    ax5.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax5.set_ylabel("Amplitude Ratio", fontsize=FONT_LABEL)
    ax5.set_xlim(0, F_MAX_PLOT)
    ax5.tick_params(labelsize=FONT_TICK)
    ax5.grid(True, linestyle='--', alpha=0.5)
    ax5.legend(fontsize=FONT_LEGEND)
    fig5.tight_layout()

    # ======================================================
    # 图6：Group1 频段相对能量
    # ======================================================
    bands_labels = [f"{b[0]}-{b[1]} Hz" for b in FREQ_BANDS]
    relative_contents = []
    for rec in data_records1:
        freq = rec["freq_pos"]
        amp = rec["amp_pos"]
        mask_total = (freq >= 0) & (freq <= 100)
        total_amp = np.sum(amp[mask_total])
        rel = []
        for low, high in FREQ_BANDS:
            mask_band = (freq >= low) & (freq <= high)
            band_amp = np.sum(amp[mask_band])
            rel.append(band_amp / total_amp)
        relative_contents.append(rel)

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    x_indexes = np.arange(len(FREQ_BANDS))
    total_width = 0.7
    bar_width = total_width / num_files
    for idx in range(num_files):
        current_x = x_indexes - total_width/2 + (idx + 0.5)*bar_width
        ax6.bar(current_x, relative_contents[idx], width=bar_width, label=LABELS_GROUP1[idx])
    ax6.set_title("Relative Frequency Band Content (Group 1)", fontsize=FONT_TITLE)
    ax6.set_xticks(x_indexes)
    ax6.set_xticklabels(bands_labels, fontsize=FONT_TICK)
    ax6.set_ylabel("Relative Content", fontsize=FONT_LABEL)
    ax6.tick_params(labelsize=FONT_TICK)
    ax6.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax6.legend(fontsize=FONT_LEGEND)
    fig6.tight_layout()

    # ======================================================
    # 图7：Group1 单独波形子图
    # ======================================================
    fig7, axes7 = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=False)
    axes7 = axes7.flatten()
    for i, rec in enumerate(data_records1):
        ax = axes7[i]
        ax.plot(rec['t_fft'], rec['trace_fft'], linewidth=1.5)
        ax.set_title(rf'$\phi$={LABELS_GROUP1[i]}', fontsize=FONT_TITLE)
        ax.set_xlim(0.15, 0.55)
        ax.set_ylim(-0.24, 0.22)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=FONT_TICK)
        ax.text(0.5, -0.23, f'({chr(97+i)})', transform=ax.transAxes, ha='center', va='top', fontsize=14)
        # if i >= 3:
        #     ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
        # if i % 3 == 0:
        #     ax.set_ylabel("Amplitude", fontsize=FONT_LABEL)
        ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
        ax.set_ylabel("Amplitude", fontsize=FONT_LABEL)
    fig7.suptitle("Time Domain", fontsize=16)
    fig7.tight_layout()

    # ======================================================
    # 图8：波形对比 (Group1 vs Group2)
    # ======================================================
    fig8, axes8 = plt.subplots(2, 3, figsize=(16, 9), sharex=False, sharey=False)
    axes8 = axes8.flatten()
    for i in range(num_files):
        ax = axes8[i]
        ax.plot(data_records1[i]['t_fft'], data_records1[i]['trace_fft'],
                linewidth=1.5, label=GROUP1_NAME)
        ax.plot(group2_data[i]['t_fft'], group2_data[i]['trace_fft'],
                linewidth=1.5, linestyle='--', label=GROUP2_NAME)
        ax.set_title(f"$\phi$={LABELS_GROUP1[i]}", fontsize=FONT_TITLE)
        ax.set_xlim(0.15, 0.55)
        ax.set_ylim(-0.24, 0.22)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(labelsize=FONT_TICK)
        ax.text(0.5, -0.23, f'({chr(97+i)})', transform=ax.transAxes, ha='center', va='top', fontsize=14)
        ax.legend(fontsize=9)
        # if i >= 3:
        #     ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
        # if i % 3 == 0:
        #     ax.set_ylabel("Amplitude", fontsize=FONT_LABEL)
        ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
        ax.set_ylabel("Amplitude", fontsize=FONT_LABEL)
    fig8.suptitle("Time Domain", fontsize=16)
    fig8.tight_layout()

    # ======================================================
    # 图9：频谱对比 (Group1 vs Group2)
    # ======================================================
    fig9, axes9 = plt.subplots(2, 3, figsize=(16, 9), sharex=False, sharey=False)
    axes9 = axes9.flatten()
    for i in range(num_files):
        ax = axes9[i]
        ax.plot(data_records1[i]['freq_pos'], data_records1[i]['amp_pos'],
                linewidth=1.5, label=GROUP1_NAME)
        ax.plot(group2_data[i]['freq_pos'], group2_data[i]['amp_pos'],
                linewidth=1.5, linestyle='--', label=GROUP2_NAME)
        ax.set_title(f"$\phi$={LABELS_GROUP1[i]}", fontsize=FONT_TITLE)
        ax.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
        ax.set_ylabel("Amplitude", fontsize=FONT_LABEL)
        ax.set_xlim(0, F_MAX_PLOT)
        ax.set_ylim(0.0, 3.72)
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.text(0.5, -0.23, f'({chr(97+i)})', transform=ax.transAxes, ha='center', va='top', fontsize=14)
        ax.legend(fontsize=9)
    fig9.suptitle("Amplitude Spectra", fontsize=16)
    fig9.tight_layout()

    plt.show()

# ==========================================================
# 运行
# ==========================================================

if __name__ == '__main__':
    main()