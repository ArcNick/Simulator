import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 用户配置 ====================

NX = 147
DTYPE = np.float32
BASE_DIR = "."
SAVE_FIG = False

PAIRS = [
    ("10poro.bin", "0poro.bin", "Reservoir Effect\n(e=0.10)"),
    ("5poro.bin",  "0poro.bin", "Reservoir Effect\n(e=0.05)"),
    ("10.bin",     "0.bin",     "Fracture Effect\n(e=0.10)"),
    ("5.bin",      "0.bin",     "Fracture Effect\n(e=0.05)")
]

# ================================================


def load_bin(filepath, nx, dtype):

    data = np.fromfile(filepath, dtype=dtype)

    if data.size % nx != 0:
        raise ValueError(
            f"{filepath} 无法reshape成(*,{nx})"
        )

    ny = data.size // nx

    return data.reshape((ny, nx))


def normalized_difference(a, b):

    diff = a - b

    return (
        np.linalg.norm(diff)
        /
        np.linalg.norm(b)
    )


def rms_difference(a, b):

    diff = a - b

    return np.sqrt(
        np.mean(diff**2)
    )


def main():

    labels = []
    norm_ratios = []
    rms_vals = []

    print("\n============================")
    print("Normalized Difference Result")
    print("============================\n")

    for file1, file2, label in PAIRS:

        data1 = load_bin(
            os.path.join(BASE_DIR, file1),
            NX,
            DTYPE
        )

        data2 = load_bin(
            os.path.join(BASE_DIR, file2),
            NX,
            DTYPE
        )

        start_sample = 120

        data1 = data1[start_sample:, :]
        data2 = data2[start_sample:, :]

        R = normalized_difference(
            data1,
            data2
        )

        rms = rms_difference(
            data1,
            data2
        )

        labels.append(label)
        norm_ratios.append(R)
        rms_vals.append(rms)

        print(
            f"{label:20s}"
            f"  R = {R:.4f}"
            f" ({R*100:.2f}%)"
            f"   RMS = {rms:.4e}"
        )

    # ==================================
    # 画图
    # ==================================

    fig, ax = plt.subplots(
        figsize=(8,5)
    )

    bars = ax.bar(
        labels,
        np.array(norm_ratios)*100
    )

    ax.set_ylabel(
        "Normalized L2 Difference (%)"
    )

    # ax.set_title(
    #     r"$R=||u-u_0||/||u_0||$"
    # )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    for bar, val in zip(
        bars,
        norm_ratios
    ):
        ax.text(
            bar.get_x()+bar.get_width()/2,
            bar.get_height(),
            f"{val*100:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(
            "normalized_difference_bar.png",
            dpi=300
        )

    plt.show()


if __name__ == "__main__":
    main()