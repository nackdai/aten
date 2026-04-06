import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
n_values = {1.331: 'red', 1.335: 'green', 1.341: 'blue'}
y = np.linspace(0.7, 1, 1000)

fig, ax = plt.subplots(figsize=(12, 8))

for n, color in n_values.items():
    # 主虹 (Primary)
    theta1 = np.degrees(4 * np.arcsin(y / n) - 2 * np.arcsin(y))
    # 副虹 (Secondary)
    theta2 = np.degrees(np.pi + 2 * np.arcsin(y) - 6 * np.arcsin(y / n))
    
    # プロット
    ax.plot(y, theta1, color=color, lw=2, label=f'Primary (n={n})')
    ax.plot(y, theta2, color=color, lw=1.5, ls='--', label=f'Secondary (n={n})')

# --- 縦軸（Y軸）を右側に設定 ---
ax.yaxis.tick_right()            # 目盛りを右側に
ax.yaxis.set_label_position("right")  # ラベルを右側に

# グラフの装飾
ax.set_title('Rainbow Scattering Angles (Y-axis on the right)', fontsize=14, pad=20)
ax.set_xlabel('Impact Parameter (y)', fontsize=12)
ax.set_ylabel('Angle (degrees)', fontsize=12)

# 虹の角度付近を拡大
ax.set_ylim(30, 60)
ax.grid(True, which='both', linestyle=':', alpha=0.6)

# 凡例
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', fontsize=9, ncol=2)

plt.tight_layout()
plt.show()