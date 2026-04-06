import matplotlib.pyplot as plt
import math

from airy_function import compute_h, compute_theta_max, compute_z, compute_M, airy_rainbow_integral

# 1. 積分計算の精度を上げる
def airy_rainbow_integral(z, u_max=10.0, du=0.005):
    """
    u_max: 振動が激しすぎるため、実用的な範囲（10程度）に留めるのがコツです
    du: より細かくすることで滑らかになります
    """
    f_z = 0.0
    n_steps = int(u_max / du)
    for i in range(n_steps + 1):
        u = i * du
        # 式(2): cos(pi/2 * (u^3 - z*u))
        phase = (math.pi / 2.0) * (u**3 - z * u)
        val = math.cos(phase)
        
        if i == 0 or i == n_steps:
            f_z += 0.5 * val
        else:
            f_z += val
            
    return f_z * du

# The following min/max values comes from the heuristic observation of the actual calculations.
Z_MIN = -40.0
Z_MAX = 25.0
Z_STEP = 0.05

def table_airy_rainbow_integral(u_max=10.0, du=0.005) -> list[float]:
    """
    u_max: 振動が激しすぎるため、実用的な範囲（10程度）に留めるのがコツです
    du: より細かくすることで滑らかになります
    """
    table = []
    z = Z_MIN
    while z <= Z_MAX:
        f_z = airy_rainbow_integral(z, u_max, du)
        table.append(f_z)
        z += Z_STEP 
    return table

def pick_airy_rainbow_integral(table: list[float], z: float) -> float:
    """
    table: 事前に計算された積分値のテーブル
    z: 積分の引数
    u_max: 振動が激しすぎるため、実用的な範囲（10程度）に留めるのがコツです
    du: テーブルのステップ幅
    """
    _z = min(Z_MAX - Z_MIN, z - Z_MIN)
    index = int(_z / Z_STEP)
    next_index = index + 1
    r = _z % Z_STEP

    if index < 0:
        return 0.0
    elif next_index >= len(table):
        return table[-1]
    else:
        # 線形補間
        return table[index] * (1 - r) + table[next_index] * (r)

# 2. グラフ描画用のサンプル点 (z) を大幅に増やす
theta_values = [x * 0.01 for x in range(3600, 5000)]
f_squared_values_R = []
f_squared_values_G = []
f_squared_values_B = []

a = 0.2e-3  # 半径 (m)

# 波長ごとの屈折率 (水の一般的な値)
n_R = 1.331  # 660nm
n_G = 1.335  # 510nm
n_B = 1.343  # 430nm

# 各色ごとに定数を計算
h_R = compute_h(n_R)
tm_R = compute_theta_max(n_R)

h_G = compute_h(n_G)
tm_G = compute_theta_max(n_G)

h_B = compute_h(n_B)
tm_B = compute_theta_max(n_B)

airy_rainbow_integral_table = table_airy_rainbow_integral()

# ループ内での計算
for theta_deg in theta_values:
    theta = math.radians(theta_deg)
    
    theta = math.radians(theta_deg)
    
    # Red
    z_R = compute_z(660e-9, a, theta, h_R, tm_R)
    fz_R = pick_airy_rainbow_integral(airy_rainbow_integral_table, z_R)
    M_R = compute_M(1, 660e-9, a, theta, h_R, tm_R)
    # M_Rの2乗を掛けて強度とする
    f_squared_values_R.append((M_R**2) * (fz_R**2))
    
    # Green
    z_G = compute_z(510e-9, a, theta, h_G, tm_G)
    fz_G = pick_airy_rainbow_integral(airy_rainbow_integral_table, z_G)
    M_G = compute_M(1, 510e-9, a, theta, h_G, tm_G)
    f_squared_values_G.append((M_G**2) * (fz_G**2))

    # Blue
    z_B = compute_z(430e-9, a, theta, h_B, tm_B)
    fz_B = pick_airy_rainbow_integral(airy_rainbow_integral_table, z_B)
    M_B = compute_M(1, 430e-9, a, theta, h_B, tm_B)
    f_squared_values_B.append((M_B**2) * (fz_B**2))

# グラフの表示
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(theta_values, f_squared_values_R, color='#ff0000', lw=1.5, label='$f^2(z)$')
plt.plot(theta_values, f_squared_values_G, color='#00ff00', lw=1.5, label='$f^2(z)$')
plt.plot(theta_values, f_squared_values_B, color='#0000ff', lw=1.5, label='$f^2(z)$')

# 論文の図3の雰囲気に近づける設定
plt.title('Airy Rainbow Intensity Distribution (Refined)', fontsize=12)
plt.xlabel('θ (degrees)', fontsize=10)
plt.ylabel('Intensity $f^2(z)$', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', lw=0.8)
plt.xlim(36, 50)
plt.legend()
plt.show()