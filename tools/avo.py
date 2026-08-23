#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ===================== 配置 =====================

SHOTS = [0, 1, 2]

CUTS = {
    0: 0.10,
    1: 0.078,
    2: 0.10
}

FRACTIONS = ['0', '10', '20', '30']

DT = 0.001

SAVE = False
OUTDIR = 'images/fracture_analysis'

BROTHER_DIR = './brother'
PORO_DIR = os.path.join(BROTHER_DIR, 'poro')

# 储层中心
RESERVOIR_CENTER = 339

# 对应炮点位置
POSX = [100, 339, 578]

# ===============================================


def get_fft(trace, dt):
    n = len(trace)
    freq = np.fft.rfftfreq(n, dt)
    spec = np.abs(np.fft.rfft(trace))
    return freq, spec


def difference_metric(base, target):
    n = min(len(base), len(target))

    b = base[:n]
    t = target[:n]

    return np.linalg.norm(t - b) / np.linalg.norm(b)


def find_record_file(base_dir, frac, shot_idx):

    rec_dir = os.path.join(base_dir, frac, 'record')

    if not os.path.isdir(rec_dir):
        return None

    for fn in os.listdir(rec_dir):
        if ('vz' in fn) and (str(shot_idx) in fn):
            return os.path.join(rec_dir, fn)

    return None


def extract_mirror_trace(
        rec_file,
        nx_total,
        pml,
        posx_shot,
        cut_time):

    data = np.fromfile(rec_file, dtype=np.float32)

    nt = len(data) // nx_total

    rec = data[:nt * nx_total].reshape(nt, nx_total)

    left_idx = pml + 3

    idx_sym_global = 2 * RESERVOIR_CENTER - posx_shot

    idx_sym_c = idx_sym_global - left_idx

    nx_crop = nx_total - (pml + 3) - (pml + 4)

    if idx_sym_c < 0 or idx_sym_c >= nx_crop:
        return None

    trace = rec[:, idx_sym_c]

    start_sample = int(round(cut_time / DT))

    return trace[start_sample:].astype(np.float64)


def main():

    model = json.load(open('./models/models.json'))
    params = json.load(open('./models/params.json'))

    nx_total = int(model['coarse']['nx'])
    pml = int(params['cpml']['thickness'])

    for shot_idx in SHOTS:

        cut = CUTS[shot_idx]

        posx_shot = POSX[shot_idx]

        traces = {}

        for frac in FRACTIONS:

            rf = find_record_file(
                PORO_DIR,
                frac,
                shot_idx)

            if rf is None:
                continue

            trace = extract_mirror_trace(
                rf,
                nx_total,
                pml,
                posx_shot,
                cut)

            if trace is not None:
                traces[frac] = trace

        if '0' not in traces:
            print(f'Shot {shot_idx}: no baseline')
            continue

        base = traces['0']

        # ==================================================
        # 图
        # ==================================================

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10)
        )

        fig.suptitle(
            f'Shot {shot_idx}',
            fontsize=18,
            fontweight='bold'
        )

        # --------------------------------------------------
        # 1. 时域
        # --------------------------------------------------

        ax = axes[0, 0]

        for frac in FRACTIONS:

            if frac not in traces:
                continue

            tr = traces[frac]

            t = np.arange(len(tr)) * DT + cut

            ax.plot(
                t,
                tr,
                lw=1.5,
                label=f'{frac}%'
            )

        ax.set_title('Time Domain')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')

        ax.grid(alpha=0.3)
        ax.legend()

        # --------------------------------------------------
        # 2. 频谱
        # --------------------------------------------------

        ax = axes[0, 1]

        for frac in FRACTIONS:

            if frac not in traces:
                continue

            f, s = get_fft(
                traces[frac],
                DT
            )

            ax.plot(
                f,
                s,
                lw=1.5,
                label=f'{frac}%'
            )

        ax.set_xlim(0, 100)

        ax.set_title('Amplitude Spectrum')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')

        ax.grid(alpha=0.3)
        ax.legend()

        # --------------------------------------------------
        # 3. 差分波形
        # --------------------------------------------------

        ax = axes[1, 0]

        for frac in ['10', '20', '30']:

            if frac not in traces:
                continue

            n = min(
                len(base),
                len(traces[frac])
            )

            diff = traces[frac][:n] - base[:n]

            t = np.arange(n) * DT + cut

            ax.plot(
                t,
                diff,
                lw=1.5,
                label=f'{frac}-0'
            )

        ax.axhline(
            0,
            color='k',
            lw=0.8
        )

        ax.set_title('Difference Trace')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude Difference')

        ax.grid(alpha=0.3)
        ax.legend()

        # --------------------------------------------------
        # 4. 差异度
        # --------------------------------------------------

        ax = axes[1, 1]

        x = []
        y = []

        for frac in ['10', '20', '30']:

            if frac not in traces:
                continue

            d = difference_metric(
                base,
                traces[frac]
            )

            x.append(int(frac))
            y.append(d)

        ax.plot(
            x,
            y,
            '-o',
            lw=2
        )

        ax.set_title('Difference Metric')

        ax.set_xlabel('Fracture Sets')

        ax.set_ylabel(
            r'$||u-u_0||/||u_0||$'
        )

        ax.grid(alpha=0.3)

        plt.tight_layout()

        if SAVE:

            os.makedirs(
                OUTDIR,
                exist_ok=True
            )

            plt.savefig(
                os.path.join(
                    OUTDIR,
                    f'shot_{shot_idx}.png'
                ),
                dpi=200
            )

            plt.close()

        else:
            plt.show()


if __name__ == '__main__':
    main()