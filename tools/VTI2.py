import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

SOLID   = 0
VESOLID = 1
FLUID   = 2
# ========== 模型参数 ==========
# 粗网格尺寸
nx = 701
nz = 601
dx = 2
dz = 2

# 时间参数
fpeak = 30.0
dt = 5e-5
nt = 20000
snapshot = 400

# epsilon = [0.08, 0.1, 0.05, 0.15]
# delta = [0.03, 0.06, 0.02, 0.08]
epsilon = [0.0, 0.0, 0.0, 0.0]
delta = [0.0, 0.0, 0.0, 0.0]
rho = [2300.0, 2450.0, 2350.0, 2600.0]
vp = [3500.0, 4000.0, 3700.0, 4800.0]
vs = [1900.0, 2200.0, 2000.0, 2700.0]

C33 = np.array(rho) * np.array(vp)**2
C55 = np.array(rho) * np.array(vs)**2
C11 = C33 * (1 + 2 * np.array(epsilon))
C13 = ((C33 - C55) * (2 * C33 * np.array(delta) + (C33 - C55)))**0.5 - C55

# 震源位置
posx = [nx // 2]
posz = 40

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = 4800
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
coarse_rho = np.full((nz, nx), rho[0], dtype=np.float32)
coarse_C11 = np.full((nz, nx), C11[0], dtype=np.float32)
coarse_C13 = np.full((nz, nx), C13[0], dtype=np.float32)
coarse_C33 = np.full((nz, nx), C33[0], dtype=np.float32)
coarse_C55 = np.full((nz, nx), C55[0], dtype=np.float32)
coarse_zeta = np.full((nz, nx), 0, dtype=np.float32)
coarse_taup = np.full((nz, nx), 0, dtype=np.float32)
coarse_taus = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig1 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig2 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig3 = np.full((nz, nx), 0, dtype=np.float32)

coarse_rho[120:280, :] = rho[1]
coarse_C11[120:280, :] = C11[1]
coarse_C13[120:280, :] = C13[1]
coarse_C33[120:280, :] = C33[1]
coarse_C55[120:280, :] = C55[1]

coarse_rho[280:450, :] = rho[2]
coarse_C11[280:450, :] = C11[2]
coarse_C13[280:450, :] = C13[2]
coarse_C33[280:450, :] = C33[2]
coarse_C55[280:450, :] = C55[2]

coarse_rho[450:, :] = rho[3]
coarse_C11[450:, :] = C11[3]
coarse_C13[450:, :] = C13[3]
coarse_C33[450:, :] = C33[3]
coarse_C55[450:, :] = C55[3]

# ========== 粗网格可视化：纵波阻抗 ==========
coarse_imp = np.sqrt(coarse_C33 * coarse_rho)  # 纵波阻抗
plt.figure(figsize=(12, 10))
plt.imshow(coarse_imp, cmap='viridis', aspect='auto', origin='upper')
plt.colorbar(label='Impedance (kg/(m²·s))')
plt.title('Coarse Model: P-wave Impedance')
plt.xlabel('X (grid cells)')
plt.ylabel('Z (grid cells)')
plt.tight_layout()
plt.savefig(os.path.join(coarse_dir, "impedance.png"), dpi=150)
plt.close()
print("粗网格阻抗图已保存至 models/coarse/impedance.png")


# ========== 生成 models.json ==========
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
