import os
import json
import numpy as np
import sls
import sys
import schoenberg as sch
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 材质常量
SOLID    = 0
VESOLID  = 1
FLUID    = 2

# 多炮控制开关
MULTISHOT = True
FLUID_POROSITY = True
FINE_SLS = True
# =============================================================================
#                               主 函 数
# =============================================================================
def main():
    
    # ---------------------------------------------------------
    # 1. 基础网格与物理参数
    # ---------------------------------------------------------
    nz, nx = 706, 691
    dx, dz = 1.5, 1.5

    fpeak    = 30.0
    dt       = 4e-6
    nt       = 132500
    snapshot = 4000

    # ---------------------------------------------------------
    # 2. 读入速度模型并计算刚度参数
    # ---------------------------------------------------------
    input_vp = "Vp_brother.bin"
    if not os.path.exists(input_vp):
        print(f"错误: 未找到输入文件 {input_vp}，请检查路径。")
        return

    vp = np.fromfile(input_vp, dtype=np.float32).reshape((nz, nx))
    vp_max = vp.max()
    vs = vp / 1.83

    epsilon = 0
    delta   = 0

    # 各向同性/各向异性刚度计算
    coarse_rho = 280 * np.power(vp, 0.265)
    coarse_C33 = coarse_rho * vp**2
    coarse_C55 = coarse_rho * vs**2
    coarse_C11 = coarse_C33 * (1 + 2 * epsilon)
    coarse_C13 = ((coarse_C33 - coarse_C55) * (2 * coarse_C33 * delta + (coarse_C33 - coarse_C55)))**0.5 - coarse_C55
    
    # ---------------------------------------------------------
    # 3. 粘弹性标准线性固体 (SLS) 衰减参数
    # ---------------------------------------------------------
    Qp = 40
    Qs = 25
    sls_params = sls.get_sls_parameters(Qp, Qs, 3, 1, 100)
    inv_tss    = 1 / sls_params["tau_sigmas"]
    taup       = sls_params["taup"]
    taus       = sls_params["taus"]

    # ---------------------------------------------------------
    # 4. 细网格流体属性
    # --------------------------------------------------------- 

    rho_fluid = 850
    vp_rho = 1300
    vs_rho = 0
    C11_fluid = rho_fluid * vp_rho**2
    C13_fluid = C11_fluid
    C33_fluid = C11_fluid
    C55_fluid = 0
    Qp_fluid = 40
    zeta = C11_fluid / (2 * np.pi * fpeak * Qp_fluid)

    border = 8
    phi_target = 0.15
    sigma_x = 0.8
    sigma_y = 0.8
    seed = 148

    # ---------------------------------------------------------
    # 5. 观测系统 (观测炮点设置)
    # ---------------------------------------------------------
    NUM_SHOTS = 10
    SHOT_INTERVAL = 50
    basex = nx // 2
    if MULTISHOT:
        # start_offset = -(NUM_SHOTS // 2 - 0.5) * SHOT_INTERVAL
        posx = np.linspace(51, 639, 147).astype(int).tolist()
        print(posx)
        # posx = [100, 339, 578]
    else:
        posx = [basex]
    posz = 38

    # CPML 吸收边界参数
    cpml_thickness = 20
    cpml_N         = 3
    cp_max         = float(vp_max)
    Rc             = 0.0001
    kappa0         = 1.2

    # ---------------------------------------------------------
    # 6. 初始化粗网格属性矩阵
    # ---------------------------------------------------------
    coarse_MAT        = np.full((nz, nx), SOLID, dtype=np.int32)
    coarse_zeta       = np.full((nz, nx), 0,     dtype=np.float32)
    coarse_taup       = np.full((nz, nx), 0,     dtype=np.float32)
    coarse_taus       = np.full((nz, nx), 0,     dtype=np.float32)
    coarse_inv_tsig1  = np.full((nz, nx), 0,     dtype=np.float32)
    coarse_inv_tsig2  = np.full((nz, nx), 0,     dtype=np.float32)
    coarse_inv_tsig3  = np.full((nz, nx), 0,     dtype=np.float32)

    targets = [5500, 5550, 5580, 5630, 5750]
    coarse_MAT[np.isin(np.round(vp).astype(int), targets)] = VESOLID
    coarse_MAT[-40:, :] = SOLID

    base_frac = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # variance = base_frac // 3
    for iz in range(nz):
        for ix in range(nx):
            if vp[iz, ix] in targets:
                num_frac = np.random.uniform(0.8, 1) * base_frac
                coarse_C11[iz, ix], coarse_C13[iz, ix], coarse_C33[iz, ix], coarse_C55[iz, ix] = sch.single_fracture_HTI(
                    coarse_C11[iz, ix], coarse_C13[iz, ix], coarse_C33[iz, ix], coarse_C55[iz, ix], 
                    z_n=7.38e-11, z_t=2.18e-11, num_frac=num_frac / 100
                )
    
    coarse_taup[coarse_MAT == VESOLID]      = taup
    coarse_taus[coarse_MAT == VESOLID]      = taus
    coarse_inv_tsig1[coarse_MAT == VESOLID] = inv_tss[0]
    coarse_inv_tsig2[coarse_MAT == VESOLID] = inv_tss[1]
    coarse_inv_tsig3[coarse_MAT == VESOLID] = inv_tss[2]
    
    # 创建输出文件夹路径
    base_dir   = "models"
    coarse_dir = os.path.join(base_dir, "coarse")
    fine_dir   = os.path.join(base_dir, "fine")
    os.makedirs(coarse_dir, exist_ok=True)
    os.makedirs(fine_dir, exist_ok=True)


    # ---------------------------------------------------------
    # 7. 细网格局部加密区块处理
    # ---------------------------------------------------------
    fine_regions = [
        {
            "x_start": 330, "x_end": 349, 
            "z_start": 531, "z_end": 550, 
            "N": 15
        }
    ]
    
    coarse_dict = {
        'rho': coarse_rho, 'C11': coarse_C11, 'C13': coarse_C13,
        'C33': coarse_C33, 'C55': coarse_C55, 'taup': coarse_taup,
        'taus': coarse_taus, 'inv_tsig1': coarse_inv_tsig1,
        'inv_tsig2': coarse_inv_tsig2, 'inv_tsig3': coarse_inv_tsig3,
        'zeta': coarse_zeta, 'MAT': coarse_MAT,
    }

    fine_list = []
    for idx, region in enumerate(fine_regions):
        x_start, x_end = region["x_start"], region["x_end"]
        z_start, z_end = region["z_start"], region["z_end"]
        N = region["N"]
    
        lenx = (x_end - x_start) * N + 1
        lenz = (z_end - z_start) * N + 1

        # 构建细网格
        fine = build_fine_grid_from_coarse(
            coarse_dict, x_start, x_end, z_start, z_end, N
        )
        porosity_mask = generate_porosity_mask(
            shape=(lenz, lenx),
            border=border,
            phi_target=phi_target,
            sigma=(sigma_y, sigma_x), 
            seed=seed + idx
        )
        if FINE_SLS == True:
            fine['MAT'][2:-2, 2:-2] = VESOLID
            fine['taup'][2:-2, 2:-2] = taup
            fine['taus'][2:-2, 2:-2] = taus
            fine['inv_tsig1'][2:-2, 2:-2] = inv_tss[0]
            fine['inv_tsig2'][2:-2, 2:-2] = inv_tss[1]
            fine['inv_tsig3'][2:-2, 2:-2] = inv_tss[2]

        if FLUID_POROSITY == True:
            fine['MAT'][porosity_mask == 1] = FLUID
            fine['rho'][porosity_mask == 1] = rho_fluid
            fine['C11'][porosity_mask == 1] = C11_fluid
            fine['C13'][porosity_mask == 1] = C13_fluid
            fine['C33'][porosity_mask == 1] = C33_fluid
            fine['C55'][porosity_mask == 1] = C55_fluid
            fine['zeta'][porosity_mask == 1] = zeta

        # 创建并导出各个细网格二进制块文件
        region_dir = os.path.join(fine_dir, str(idx))
        os.makedirs(region_dir, exist_ok=True)

        for key in coarse_dict.keys():
            filename = f"material.bin" if key == 'MAT' else f"{key}.bin"
            fine[key].tofile(os.path.join(region_dir, filename))

        # 保存细网格质点密度可视化结果
        visualize(fine['rho'], nz=lenz, nx=lenx, dx=dx/N, dz=dz/N, 
                  save_path=os.path.join(region_dir, "rho.png"))
        # coarse_C11[z_start+1:z_end, x_start+1:x_end] = 0
        # coarse_C13[z_start+1:z_end, x_start+1:x_end] = 0
        # coarse_C33[z_start+1:z_end, x_start+1:x_end] = 0
        # coarse_C55[z_start+1:z_end, x_start+1:x_end] = 0
        # 记录到细网格索引配置
        fine_list.append({
            "x_start": x_start, "x_end": x_end, "z_start": z_start, "z_end": z_end, "N": N,
            "rho": f"models/fine/{idx}/rho.bin",
            "C11": f"models/fine/{idx}/C11.bin",
            "C13": f"models/fine/{idx}/C13.bin",
            "C33": f"models/fine/{idx}/C33.bin",
            "C55": f"models/fine/{idx}/C55.bin",
            "zeta": f"models/fine/{idx}/zeta.bin",
            "taup": f"models/fine/{idx}/taup.bin",
            "taus": f"models/fine/{idx}/taus.bin",
            "inv_tsig1": f"models/fine/{idx}/inv_tsig1.bin",
            "inv_tsig2": f"models/fine/{idx}/inv_tsig2.bin",
            "inv_tsig3": f"models/fine/{idx}/inv_tsig3.bin",
            "material": f"models/fine/{idx}/material.bin"            
        })
    visualize(coarse_C11, save_path=os.path.join('.', 'vp_brother.png'))
    # ---------------------------------------------------------
    # 8. 导出 JSON 配置文件与粗网格二进制模型
    # ---------------------------------------------------------
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
        "fine": fine_list
    }
    
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

    with open(os.path.join(base_dir, "models.json"), "w") as f:
        json.dump(models_config, f, indent=2)
    print("models.json 已生成")

    for key in coarse_dict.keys():
        filename = f"material.bin" if key == 'MAT' else f"{key}.bin"
        coarse_dict[key].tofile(os.path.join(coarse_dir, filename))
        
    print("粗网格模型二进制数据已写入完成。")


# =============================================================================
#                               可视化工具函数
# =============================================================================
def visualize(data, nz=706, nx=691, dx=1.5, dz=1.5, cmap='jet', save_path=None):
    """绘制二维切片图"""
    fig, ax = plt.subplots(figsize=(12, 9))
    extent = [0, nx, nz, 0]  # 将刻度对应到真实的公里/米空间大小
    
    im = ax.imshow(data, cmap=cmap, aspect='auto', extent=extent)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Value', fontsize=12)
    
    ax.set_title('Model Profile', fontsize=14, pad=20)
    ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
    ax.set_ylabel('Depth (m)', fontsize=12)
    
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图像已保存至: {save_path}")
    
    plt.close(fig)


# =============================================================================
#                               细网格插值函数
# =============================================================================
def build_fine_grid_from_coarse(coarse_arrays, x_start, x_end, z_start, z_end, N):
    """从粗网格数据切片并一阶线性插值放大至细网格"""
    lenx = (x_end - x_start) * N + 1
    lenz = (z_end - z_start) * N + 1

    sub = {}
    for key, arr in coarse_arrays.items():
        sub[key] = arr[z_start:z_end+1, x_start:x_end+1]

    zoom_z = lenz / sub['rho'].shape[0]
    zoom_x = lenx / sub['rho'].shape[1]

    fine = {}
    float_keys = [
        'rho', 'C11', 'C13', 'C33', 'C55', 'taup', 'taus', 
        'inv_tsig1', 'inv_tsig2', 'inv_tsig3', 'zeta'
    ]
    
    for key in float_keys:
        fine[key] = zoom(sub[key], (zoom_z, zoom_x), order=1)

    fine['MAT'] = zoom(sub['MAT'], (zoom_z, zoom_x), order=0).astype(np.int32)

    return fine

def generate_porosity_mask(shape, border, phi_target, sigma=(2.0,1.2), seed=None):
    """
    生成二值掩膜，内部区域孔隙度=phi_target，边界border内为0。
    shape: (nz, nx) 整个区域的尺寸
    border: 边界宽度（格）
    sigma: (sigma_z, sigma_x) 高斯滤波各向异性半径
    """
    nz_total, nx_total = shape
    inner_nz = nz_total - 2 * border
    inner_nx = nx_total - 2 * border
    if inner_nz <= 0 or inner_nx <= 0:
        raise ValueError(f"border过大: border={border}")
    
    if seed is not None:
        np.random.seed(seed)
    
    noise = np.random.randn(inner_nz, inner_nx)
    smoothed = gaussian_filter(noise, sigma=sigma, mode='reflect')
    threshold = np.percentile(smoothed, phi_target * 100)
    inner_binary = (smoothed <= threshold).astype(np.uint8)
    
    full_binary = np.zeros((nz_total, nx_total), dtype=np.uint8)
    full_binary[border:nz_total-border, border:nx_total-border] = inner_binary
    return full_binary

if __name__ == '__main__':
    main()