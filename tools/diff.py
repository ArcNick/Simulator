#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 配置
# ==========================================================

SHOT = 0

BROTHER_DIR = "./brother/better"

FRACTIONS = ["0", "10", "20", "30"]

SAVE = False
OUTDIR = "./images/difference_gather"

DT = 0.001

# ==========================================================
# 读取模型参数
# ==========================================================

with open("./models/models.json", "r") as f:
    models = json.load(f)

with open("./models/params.json", "r") as f:
    params = json.load(f)

nx_total = int(models["coarse"]["nx"])

cpml = int(params["cpml"]["thickness"])

left_cut = cpml + 3
right_cut = cpml + 4

# ==========================================================
# 读炮集
# ==========================================================

def read_record(frac, shot):

    fn = os.path.join(
        BROTHER_DIR,
        frac,
        "record",
        f"record_vz_{shot}.bin"
    )

    if not os.path.exists(fn):
        raise FileNotFoundError(fn)

    data = np.fromfile(fn, dtype=np.float32)

    nt = len(data) // nx_total

    rec = data[:nt * nx_total]

    rec = rec.reshape(nt, nx_total)

    rec = rec[
        :,
        left_cut:nx_total-right_cut
    ]

    return rec

# ==========================================================
# 主程序
# ==========================================================

records = {}

for frac in FRACTIONS:

    records[frac] = read_record(
        frac,
        SHOT
    )

base = records["0"]

# ==========================================================
# 差值炮集
# ==========================================================

diff10 = records["10"] - base
diff20 = records["20"] - base
diff30 = records["30"] - base

# ==========================================================
# 差异能量
# ==========================================================

energy10 = np.sum(diff10**2, axis=0)
energy20 = np.sum(diff20**2, axis=0)
energy30 = np.sum(diff30**2, axis=0)

energy10 /= np.max(energy10)
energy20 /= np.max(energy20)
energy30 /= np.max(energy30)

# ==========================================================
# 统一色标
# ==========================================================

vmax_record = np.percentile(
    np.abs(base),
    99
)

vmax_diff = max(
    np.percentile(np.abs(diff10), 99),
    np.percentile(np.abs(diff20), 99),
    np.percentile(np.abs(diff30), 99)
)

# ==========================================================
# 时间轴
# ==========================================================

nt, nx = base.shape

tmax = nt * DT

extent = [0, nx, tmax, 0]

# ==========================================================
# 绘图
# ==========================================================

fig = plt.figure(
    figsize=(12, 8)
)

# ----------------------------------------------------------
# 0%
# ----------------------------------------------------------

ax = plt.subplot(231)

im0 = ax.imshow(
    records["0"],
    cmap="seismic",
    aspect="auto",
    vmin=-vmax_record,
    vmax=vmax_record,
    extent=extent
)

ax.set_title("0 Fracture")
ax.set_xlabel("Trace")
ax.set_ylabel("Time (s)")

cbar = plt.colorbar(im0, ax=ax)
cbar.set_label("Amplitude")

# ----------------------------------------------------------
# 30%
# ----------------------------------------------------------

ax = plt.subplot(232)

im1 = ax.imshow(
    records["30"],
    cmap="seismic",
    aspect="auto",
    vmin=-vmax_record,
    vmax=vmax_record,
    extent=extent
)

ax.set_title("30 Fracture")
ax.set_xlabel("Trace")
ax.set_ylabel("Time (s)")

cbar = plt.colorbar(im1, ax=ax)
cbar.set_label("Amplitude")

# ----------------------------------------------------------
# 10-0
# ----------------------------------------------------------

ax = plt.subplot(233)

im2 = ax.imshow(
    diff10,
    cmap="seismic",
    aspect="auto",
    vmin=-vmax_diff,
    vmax=vmax_diff,
    extent=extent
)

ax.set_title("10 - 0")
ax.set_xlabel("Trace")
ax.set_ylabel("Time (s)")

cbar = plt.colorbar(im2, ax=ax)
cbar.set_label("Amplitude Difference")

# ----------------------------------------------------------
# 20-0
# ----------------------------------------------------------

ax = plt.subplot(234)

im3 = ax.imshow(
    diff20,
    cmap="seismic",
    aspect="auto",
    vmin=-vmax_diff,
    vmax=vmax_diff,
    extent=extent
)

ax.set_title("20 - 0")
ax.set_xlabel("Trace")
ax.set_ylabel("Time (s)")

cbar = plt.colorbar(im3, ax=ax)
cbar.set_label("Amplitude Difference")

# ----------------------------------------------------------
# 30-0
# ----------------------------------------------------------

ax = plt.subplot(235)

im4 = ax.imshow(
    diff30,
    cmap="seismic",
    aspect="auto",
    vmin=-vmax_diff,
    vmax=vmax_diff,
    extent=extent
)

ax.set_title("30 - 0")
ax.set_xlabel("Trace")
ax.set_ylabel("Time (s)")

cbar = plt.colorbar(im4, ax=ax)
cbar.set_label("Amplitude Difference")

# ----------------------------------------------------------
# 差异能量
# ----------------------------------------------------------

ax = plt.subplot(236)

x = np.arange(nx)

ax.plot(
    x,
    energy10,
    lw=2,
    label="10-0"
)

ax.plot(
    x,
    energy20,
    lw=2,
    label="20-0"
)

ax.plot(
    x,
    energy30,
    lw=2,
    label="30-0"
)

imax10 = np.argmax(energy10)
imax20 = np.argmax(energy20)
imax30 = np.argmax(energy30)

ax.axvline(imax10,
           ls="--",
           alpha=0.5)

ax.axvline(imax20,
           ls="--",
           alpha=0.5)

ax.axvline(imax30,
           ls="--",
           alpha=0.5)

ax.set_title("Difference Energy")
ax.set_xlabel("Trace")
ax.set_ylabel("Normalized Energy")

ax.grid(True)
ax.legend()

# ==========================================================
# 输出
# ==========================================================

plt.suptitle(
    f"Shot {SHOT}",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

if SAVE:

    os.makedirs(
        OUTDIR,
        exist_ok=True
    )

    outfile = os.path.join(
        OUTDIR,
        f"shot{SHOT}_difference.png"
    )

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    print(outfile)

else:

    plt.show()