import numpy as np
import matplotlib.pyplot as plt
import sls
import os
import json
import schoenberg as sch
SOLID = 0
VESOLID = 1
FLUID = 2

def visualize_vp(vp, nz=706, nx=691, dx=1.5, dz=1.5, cmap='jet', save_path=None):
    plt.figure(figsize=(12, 9))
    extent = [0, nx, nz, 0]
    im = plt.imshow(vp, cmap=cmap, aspect='auto', extent=extent)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Velocity (m/s)', fontsize=12)
    
    plt.title(f'Velocity Model ', fontsize=14, pad=20)
    plt.xlabel('Horizontal Distance (m)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图像已保存至: {save_path}")
    
    plt.show()

# --- 参数设置 ---
nz, nx = 706, 690  # 原始尺寸
dx = 1.5               # 目标间距
dz = 1.5

# 时间参数
fpeak = 30.0
dt = 1e-5
nt = 60000
snapshot = 400

input_file = "vp.bin"

vp = np.fromfile(input_file, dtype=np.float32)
vp = vp.reshape((nz, nx))
nx = 691
vp = np.hstack([vp, vp[:, -1].reshape(-1, 1)])
# for iz in range(nz):
#     for ix in range(nx):
#         if vp[iz, ix] == 5750: vp[iz, ix] = 6350
#         if vp[iz, ix] == 5500: vp[iz, ix] = 6000
#         if vp[iz, ix] == 5630: vp[iz, ix] = 6200
#         if vp[iz, ix] == 5550: vp[iz, ix] = 6200
#         if vp[iz, ix] == 5580: vp[iz, ix] = 5860
# for iz in range(272, 370):
#     for ix in range(281, 304):
#         vp[iz, ix] = 5860
# for iz in range(290, 384):
#     for ix in range(419, 443):
#         vp[iz, ix] = 5860
# for iz in range(144, 272):
#     for ix in range(262, 284):
#         vp[iz, ix] = 6000
# for iz in range(144, 291):
#     for ix in range(440, 458):
#         vp[iz, ix] = 6000

vp[79, 180] = 4800
vp[79, 181] = 4800
vp[78, 180] = 4800
vp[78, 181] = 4800
vp[561:580, 350:370] = 6350
vp_max = vp.max()
vs = vp / 1.89
coarse_rho = 310 * np.power(vp, 0.25)

epsilon_1 = 0.0
delta_1 = 0.0
coarse_C33 = coarse_rho * vp**2
coarse_C55 = coarse_rho * vs**2
coarse_C11 = coarse_C33 * (1 + 2 * epsilon_1)
coarse_C13 = ((coarse_C33 - coarse_C55) * (2 * coarse_C33 * delta_1 + (coarse_C33 - coarse_C55)))**0.5 - coarse_C55

Qp1 = 25
Qs1 = 15
sls_params = sls.get_sls_parameters(Qp1, Qs1, 3, 2, 100)
inv_tss1 = 1 / sls_params["tau_sigmas"]
taup1 = sls_params["taup"]
taus1 = sls_params["taus"]

Qp2 = 40
Qs2 = 25
sls_params = sls.get_sls_parameters(Qp2, Qs2, 3, 2, 100)
inv_tss2 = 1 / sls_params["tau_sigmas"]
taup2 = sls_params["taup"]
taus2 = sls_params["taus"]

# 震源位置
posx = nx // 2
posz = 38

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = 6500
Rc = 0.0001
kappa0 = 1.2

# ========== 创建目录结构 ==========
base_dir = "models"
coarse_dir = os.path.join(base_dir, "coarse")
fine_dir = os.path.join(base_dir, "fine")
os.makedirs(coarse_dir, exist_ok=True)
os.makedirs(fine_dir, exist_ok=True)

# ========== 生成粗网格模型 ==========
coarse_MAT = np.full((nz, nx), SOLID, dtype=np.int32)
coarse_zeta = np.full((nz, nx), 0, dtype=np.float32)
coarse_taup = np.full((nz, nx), 0, dtype=np.float32)
coarse_taus = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig1 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig2 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig3 = np.full((nz, nx), 0, dtype=np.float32)

models_config = {
    "coarse": {
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "rho": "models/coarse/rho.bin",
        "C11": "models/coarse/C11.bin",
        "C13": "models/coarse/C13.bin",
        "C33": "models/coarse/C33.bin",
        "C55": "models/coarse/C55.bin",
        "zeta": "models/coarse/zeta.bin",
        "taup": "models/coarse/taup.bin",
        "taus": "models/coarse/taus.bin",
        "inv_tsig1": "models/coarse/inv_tsig1.bin",
        "inv_tsig2": "models/coarse/inv_tsig2.bin",
        "inv_tsig3": "models/coarse/inv_tsig3.bin",
        "material": "models/coarse/material.bin",
    },
    "fine": []
}

visualize_vp(vp)

with open(os.path.join(base_dir, "models.json"), "w") as f:
    json.dump(models_config, f, indent=2)
print("models.json 已生成")

# ========== 生成 params.json ==========
params_config = {
    "base": {
        "fpeak": fpeak,
        "dt": dt,
        "nt": nt,
        "posx": posx,
        "posz": posz,
        "snapshot": snapshot
    },
    "cpml": {
        "thickness": cpml_thickness,
        "N": cpml_N,
        "cp_max": cp_max,
        "Rc": Rc,
        "kappa0": kappa0
    }
}

with open(os.path.join(base_dir, "params.json"), "w") as f:
    json.dump(params_config, f, indent=2)
print("params.json 已生成")

# ========== 保存粗网格文件 ==========
coarse_MAT.tofile(os.path.join(coarse_dir, "material.bin"))
coarse_rho.tofile(os.path.join(coarse_dir, "rho.bin"))
coarse_C11.tofile(os.path.join(coarse_dir, "C11.bin"))
coarse_C13.tofile(os.path.join(coarse_dir, "C13.bin"))
coarse_C33.tofile(os.path.join(coarse_dir, "C33.bin"))
coarse_C55.tofile(os.path.join(coarse_dir, "C55.bin"))
coarse_taup.tofile(os.path.join(coarse_dir, "taup.bin"))
coarse_taus.tofile(os.path.join(coarse_dir, "taus.bin"))
coarse_inv_tsig1.tofile(os.path.join(coarse_dir, "inv_tsig1.bin"))
coarse_inv_tsig2.tofile(os.path.join(coarse_dir, "inv_tsig2.bin"))
coarse_inv_tsig3.tofile(os.path.join(coarse_dir, "inv_tsig3.bin"))
coarse_zeta.tofile(os.path.join(coarse_dir, "zeta.bin"))

