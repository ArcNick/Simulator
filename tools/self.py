import numpy as np
import matplotlib.pyplot as plt
import os

# ========== 配置参数 ==========
nx = 691
flag = "0"
base_path = "./brother/multishot_ult/" + flag + "/record/record_vz_"
shot_pos = [51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 
            111, 115, 119, 123, 127, 131, 135, 139, 143, 147, 151, 155, 
            159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 200, 204, 
            208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 
            256, 260, 264, 268, 272, 276, 280, 284, 288, 292, 296, 300, 
            304, 308, 312, 316, 320, 324, 328, 332, 336, 340, 345, 349, 
            353, 357, 361, 365, 369, 373, 377, 381, 385, 389, 393, 397, 
            401, 405, 409, 413, 417, 421, 425, 429, 433, 437, 441, 445, 
            449, 453, 457, 461, 465, 469, 473, 477, 481, 485, 489, 494, 
            498, 502, 506, 510, 514, 518, 522, 526, 530, 534, 538, 542, 
            546, 550, 554, 558, 562, 566, 570, 574, 578, 582, 586, 590, 
            594, 598, 602, 606, 610, 614, 618, 622, 626, 630, 634, 639]
n_files = len(shot_pos)
files = [f"{base_path}{i}.bin" for i in range(n_files)]
# ========== 颜色条范围设置 ==========
CLIM_MIN = -1e-8
CLIM_MAX = 1e-8

# ========== 自动计算 nt ==========
first_file = files[0]
file_size = os.path.getsize(first_file)
nt = file_size // (nx * 4)
print(f"自动计算: nt = {nt}, nx = {nx}")
print(f"文件总数: {len(files)}, 炮点位置数: {len(shot_pos)}")

# ========== 检查数量是否匹配 ==========
n_shots = min(len(files), len(shot_pos))
files = files[:n_shots]
shot_pos = shot_pos[:n_shots]

print(f"实际处理: {n_shots} 炮")
print(f"文件顺序: {files[0]} ... {files[-1]}")

# ========== 提取 ==========
section = np.zeros((nt, n_shots), dtype=np.float32)

for i, f in enumerate(files):
    with open(f, 'rb') as fp:
        data = np.fromfile(fp, dtype=np.float32, count=nt*nx)
    data = data.reshape(nt, nx, order='C')
    section[:, i] = data[:, shot_pos[i]]
    del data
    
    if (i+1) % 10 == 0 or i+1 == n_shots:
        print(f"{i+1}/{n_shots}")

# ========== 保存 ==========
section.tofile(flag + ".bin")
print(f"保存: zero_offset.bin, 维度: {section.shape}")

# ========== 显示 ==========
if nt > 3000:
    step = nt // 3000
    section_show = section[::step, :]
    print(f"降采样显示: {section_show.shape[0]} 个采样点 (原始 {nt})")
else:
    section_show = section

plt.figure(figsize=(9, 9))
im = plt.imshow(section_show, aspect='auto', cmap='seismic', 
                extent=[50 * 1.5, 639 * 1.5, nt * 0.001, 0],
                vmin=CLIM_MIN, vmax=CLIM_MAX)
# plt.colorbar(im, label='Amplitude')
plt.xlabel('Distance (m)', fontsize=14)
plt.ylabel('Time (s)', fontsize=14)
# plt.title('10poro')
# plt.title(f'自激自收剖面 (nt={nt}, n_shots={n_shots})')
plt.tight_layout()
# plt.savefig("zero_offset.png", dpi=150)
plt.show()

print(f"\n完成！")
print(f"数据振幅范围: [{section.min():.6f}, {section.max():.6f}]")
print(nt)