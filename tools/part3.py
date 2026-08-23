#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ===================== 配置 =====================
SHOTS = [0, 1, 2]
# 每个炮点对应的截断时间（秒）
CUTS = {
    0: 0.10,
    1: 0.078,   # 示例，请按需修改
    2: 0.10    # 示例，请按需修改
}
FRACTIONS = ['0', '5', '10']
DT = 0.001
SAVE = False
OUTDIR = 'images/part3'
BROTHER_DIR = './brother'
RESERVOIR_CENTER = 339
PORO_DIR = os.path.join(BROTHER_DIR, 'new_poro')

def get_fft(trace, dt):
    n = len(trace)
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(trace))
    return freqs, spec

def find_record_file(base_dir, frac, shot_idx):
    rec_dir = os.path.join(base_dir, frac, 'record')
    if not os.path.isdir(rec_dir):
        return None
    for fn in os.listdir(rec_dir):
        if 'vz' in fn and str(shot_idx) in fn:
            return os.path.join(rec_dir, fn)
    return None

def extract_mirror_trace(rec_file, nx_total, pml, posx_shot, cut_time):
    """cut_time: 去除直达波的时间截断（秒）"""
    data = np.fromfile(rec_file, dtype=np.float32)
    nt = len(data) // nx_total
    rec = data[:nt * nx_total].reshape((nt, nx_total))
    
    left_idx = pml + 3
    idx_sym_global = 2 * RESERVOIR_CENTER - posx_shot
    idx_sym_c = idx_sym_global - left_idx
    
    if 0 <= idx_sym_c < rec.shape[1] - left_idx - (pml + 4):
        trace = rec[:, idx_sym_c].astype(np.float64)
        start_sample = int(round(cut_time / DT))
        return trace[start_sample:]
    return None

def main():
    model = json.load(open('./models/models.json'))
    params = json.load(open('./models/params.json'))
    nx_total = int(model['coarse']['nx'])
    pml = int(params.get('cpml', {}).get('thickness', 20))
    posx_list = [int(x) for x in params.get('base', {}).get('posx', [nx_total//2]*5)]

    for shot_idx in SHOTS:
        cut = CUTS.get(shot_idx, 0.08)  # 默认值以防缺失
        posx_shot = int(posx_list[shot_idx])
        poro_traces = {}

        for frac in FRACTIONS:
            rf = find_record_file(PORO_DIR, frac, shot_idx)
            trace = extract_mirror_trace(rf, nx_total, pml, posx_shot, cut) if rf else None
            if trace is not None:
                poro_traces[frac] = trace

        if not poro_traces:
            print(f"Warning: Shot {shot_idx} - no valid mirror traces found.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Poro Mirror Trace Analysis - Shot {shot_idx} (cut={cut}s)', fontsize=16, fontweight='bold')

        # 波形
        for frac, trace in poro_traces.items():
            t = np.arange(len(trace)) * DT + cut
            axes[0].plot(t, trace, label=f'frac={frac}', lw=1.5)

        # 频谱
        for frac, trace in poro_traces.items():
            f, s = get_fft(trace, DT)
            axes[1].plot(f, s, label=f'frac={frac}', lw=1.5)

        axes[0].set_title('Time Domain Mirror Traces')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        axes[1].set_title('Amplitude Spectrum')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Amplitude')
        axes[1].set_xlim(0, 100)
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        if SAVE:
            os.makedirs(OUTDIR, exist_ok=True)
            plt.savefig(os.path.join(OUTDIR, f'shot{shot_idx}_poro_only.png'), dpi=150)
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    main()