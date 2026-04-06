import matplotlib.pyplot as plt
import math

from calc_water_refraction import calc_water_refraction, DAIMON_MASAMURA_WATER_REFRACTIVE_INDEX_TABLE
from airy_function import compute_h, compute_theta_max, compute_z, compute_M, airy_rainbow_integral

A_MIN = 0.02e-3
A_MAX = 0.3e-3
A_STEP = 0.02e-3

LAMBDA_MIN = 400
LAMBDA_MAX = 700
LAMBDA_STEP = 10

THETA_MIN = math.radians(36)
THETA_MAX = math.radians(50)
THETA_STEP = math.radians(0.1)

def _table_airy_rainbow_integral(a) -> list[list[float]]:
    """
    u_max: 振動が激しすぎるため、実用的な範囲（10程度）に留めるのがコツです
    du: より細かくすることで滑らかになります
    """
    table: list[list[list[float]]] = []

    wavelength = LAMBDA_MIN
    while wavelength <= LAMBDA_MAX:
        wavelength_list = []  # lambdaの階層
        
        theta = THETA_MIN
        while theta <= THETA_MAX:
            # --- 計算処理 ---
            n = calc_water_refraction(wavelength, DAIMON_MASAMURA_WATER_REFRACTIVE_INDEX_TABLE)
            h = compute_h(n)
            theta_max = compute_theta_max(n)
            z = compute_z(wavelength * 1e-9, a, theta, h, theta_max)
            f_z = airy_rainbow_integral(z)
            # ----------------
            
            wavelength_list.append(f_z)  # thetaの値をlambdaリストに追加
            theta += THETA_STEP
            
        wavelength += LAMBDA_STEP
        table.append(wavelength_list)  # lambdaリストをメインテーブルに追加
        #print(f"Calculated for wavelength: {wavelength} nm")

    return table

def table_airy_rainbow_integral() -> list[list[list[float]]]:
    table: list[list[list[float]]] = []
    
    a = A_MIN
    while a <= A_MAX + A_STEP * 0.5:
        print(f"Calculating for radius: {a} m")
        table.append(_table_airy_rainbow_integral(a))
        a += A_STEP

    return table

def _pick_table_with_interp(
        table: list[list[float]], 
        x: float, x_range: float,
        y: float, y_range: float,
    ) -> float:
    """
    table: 事前に計算された積分値のテーブル
    x: x座標
    y: y座標
    """
    u = x * x_range - 0.5
    v = y * y_range - 0.5
    i = int(math.floor(u))
    j = int(math.floor(v))
    u -= i
    v -= j

    i0 = max(0, min(x_range - 1, i))
    i1 = max(0, min(x_range - 1, i + 1))
    j0 = max(0, min(y_range - 1, j))
    j1 = max(0, min(y_range - 1, j + 1))

    return (table[i0][j0] * (1 - u) * (1 - v) + 
            table[i0][j1] * (1 - u) * v + 
            table[i1][j0] * u * (1 - v) + 
            table[i1][j1] * u * v)


def _pick_airy_rainbow_integral(table: list[list[float]], wavelength: float, theta: float) -> float:
    """
    table: 事前に計算された積分値のテーブル
    z: 積分の引数
    u_max: 振動が激しすぎるため、実用的な範囲（10程度）に留めるのがコツです
    du: テーブルのステップ幅
    """
    wavelength_range = len(table)
    theta_range = len(table[0])

    deg = math.degrees(theta - THETA_MIN)

    wavelength_normalized = (((wavelength - LAMBDA_MIN) / LAMBDA_STEP) + 0.5) / wavelength_range
    theta_normalized = (((theta - THETA_MIN) / THETA_STEP) + 0.5) / theta_range

    result = _pick_table_with_interp(table, wavelength_normalized, wavelength_range, theta_normalized, theta_range)
    return result

def pick_airy_rainbow_integral(table: list[list[float]], a: float, wavelength: float, theta: float) -> float:
    """
    table: 事前に計算された積分値のテーブル
    a: 半径
    wavelength: 波長
    theta: 角度
    """
    a_range = len(table)
    a_normalized = (((a - A_MIN) / A_STEP) + 0.5) / a_range

    # aの階層を選択
    u = a_normalized * a_range - 0.5
    i = int(math.floor(u))
    u -= i

    i0 = max(0, min(a_range - 1, i))
    i1 = max(0, min(a_range - 1, i + 1))

    f0 = _pick_airy_rainbow_integral(table[i0], wavelength, theta)
    f1 = _pick_airy_rainbow_integral(table[i1], wavelength, theta)

    return f0 * (1 - u) + f1 * u


# 2. グラフ描画用のサンプル点 (z) を大幅に増やす
theta_values = [x * 0.01 for x in range(3600, 5000)]

f_squared_values = []
expected_f_squared_values = []

a = 0.2e-3  # 半径 (m)

airy_rainbow_integral_table = table_airy_rainbow_integral()

wavelength = 401

# ループ内での計算
for theta_deg in theta_values:
    theta = math.radians(theta_deg)
    
    f_z = pick_airy_rainbow_integral(airy_rainbow_integral_table, a, wavelength, theta)
    f_squared_values.append(f_z**2)

    n = calc_water_refraction(wavelength, DAIMON_MASAMURA_WATER_REFRACTIVE_INDEX_TABLE)
    h = compute_h(n)
    theta_max = compute_theta_max(n)
    z = compute_z(wavelength * 1e-9, a, theta, h, theta_max)
    expected_f_z = airy_rainbow_integral(z)
    expected_f_squared_values.append(expected_f_z**2)

    x = 0

# グラフの表示
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(theta_values, f_squared_values, color='#ff0000', lw=1.5, label='$f^2(z)$')
plt.plot(theta_values, expected_f_squared_values, color='#0000ff', lw=1.5, label='$f^2(z)$')

# 論文の図3の雰囲気に近づける設定
plt.title('Airy Rainbow Intensity Distribution (Refined)', fontsize=12)
plt.xlabel('θ (degrees)', fontsize=10)
plt.ylabel('Intensity $f^2(z)$', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', lw=0.8)
plt.xlim(36, 50)
plt.legend()
plt.show()