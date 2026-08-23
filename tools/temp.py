import numpy as np

# 读取两个二进制文件（按 float32 解析）
data1 = np.fromfile('VTI_0.bin', dtype=np.float32)
data2 = np.fromfile('VTI_1.bin', dtype=np.float32)

# 确保两个文件长度一致（取较短的）
min_len = min(len(data1), len(data2))
data1 = data1[:min_len]
data2 = data2[:min_len]

# 计算差值
diff = data1 - data2

# 保存为 float32 二进制文件
diff.astype(np.float32).tofile('record_vz_0.bin')

print(f"完成！")
print(f"数据点数: {min_len}")
print(f"第一个文件范围: {data1.min():.3f} ~ {data1.max():.3f}")
print(f"第二个文件范围: {data2.min():.3f} ~ {data2.max():.3f}")
print(f"差值范围: {diff.min():.6f} ~ {diff.max():.6f}")