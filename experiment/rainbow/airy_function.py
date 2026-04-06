import matplotlib.pyplot as plt
import math

def compute_h(n: float) -> float:
    """
    n: 屈折率
    """
    h = 9 / (4 * (n**2 - 1)) * math.sqrt((4 - n**2) / (n**2 - 1))
    return h

def compute_theta_max(n: float) -> float:
    """
    n: 屈折率
    """
    theta_max = 4 * math.asin(math.sqrt((4 - n**2) / (3 * n**2))) - 2 * math.asin(math.sqrt((4 - n**2) / 3))
    return theta_max

def compute_z(wavelength: float, a: float, theta: float, h: float, theta_max: float) -> float:
    """
    wavelength: 波長
    a: 半径
    theta: 角度 (ラジアン)
    h: 計算されたhの値
    theta_max: 計算されたtheta_maxの値
    """
    z = math.pow(48 / h, 1/3) * math.pow(a / wavelength, 2/3) * (theta_max - theta)
    return z

def compute_M(k: float, wavelength: float, a: float, theta: float, h: float, theta_max: float) -> float:
    """
    k: 係数
    wavelength: 波長
    a: 半径
    theta: 角度 (ラジアン)
    h: 計算されたhの値
    theta_max: 計算されたtheta_maxの値
    """
    epsilon = theta_max - theta
    cos_eps = math.cos(epsilon)
    M = 2 * k * math.pow((3 * a**2 * wavelength) / (4 * h * cos_eps), 1/3)
    return M

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

def get_refractive_index_water(vlambda_m):
    """
    vlambda_m: 波長 (メートル単位) 例: 660e-9
    """
    lambda_um = vlambda_m * 1e6  # 単位をマイクロメートルに変換
    n = 2.04274 - 6.53632 * lambda_um + 24.9501 * lambda_um**2 - 48.2565 * lambda_um**3 + 46.7152 * lambda_um**4 - 18.0191 * lambda_um**5
    return n

def main():
    # 2. グラフ描画用のサンプル点 (z) を大幅に増やす
    theta_values = [x * 0.01 for x in range(3600, 5000)]
    f_squared_values_R = []
    f_squared_values_G = []
    f_squared_values_B = []

    a = 0.2e-3  # 半径 (m)

    # 波長ごとの屈折率 (水の一般的な値)
    n_R = get_refractive_index_water(660e-9)  # 660nm
    n_G = get_refractive_index_water(510e-9)  # 510nm
    n_B = get_refractive_index_water(430e-9)  # 430nm
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

    # NOTE:
    # compute_M に渡す k の値は、波長で同じではなく、調整が必要かも.
    # 一方で、どう調整すればいいのかは不明.
    # 現在指定している値は、目測で理想形のグラフ形状に近くなるようにしているだけで根拠はない.

    # ループ内での計算
    for theta_deg in theta_values:
        theta = math.radians(theta_deg)
        
        # Red
        z_R = compute_z(660e-9, a, theta, h_R, tm_R)
        fz_R = airy_rainbow_integral(z_R)
        M_R = compute_M(1, 660e-9, a, theta, h_R, tm_R)
        # M_Rの2乗を掛けて強度とする
        f_squared_values_R.append((M_R**2) * (fz_R**2))

        # Green
        z_G = compute_z(510e-9, a, theta, h_G, tm_G)
        fz_G = airy_rainbow_integral(z_G)
        M_G = compute_M(0.95, 510e-9, a, theta, h_G, tm_G)
        f_squared_values_G.append((M_G**2) * (fz_G**2))

        # Blue
        z_B = compute_z(430e-9, a, theta, h_B, tm_B)
        fz_B = airy_rainbow_integral(z_B)
        M_B = compute_M(0.875, 430e-9, a, theta, h_B, tm_B)
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

if __name__ == "__main__":
    main()
