import matplotlib.pyplot as plt

def calc_water_refraction(wavelength_nm: float) -> float:
    """
    wavelength_nm: 波長 (ナノメートル)
    """
    x = wavelength_nm

    # nm -> μm
    x *= 1e-3

    # https://refractiveindex.info/?shelf=main&book=H2O&page=Daimon-20.0C
    n = (1 + 5.684027565E-1 / (1-5.101829712E-3 / x**2)
         + 1.726177391E-1 / (1-1.821153936E-2 / x**2)
         + 2.086189578E-2 / (1-2.620722293E-2 / x**2)
         + 1.130748688E-1 / (1-1.069792721E1 / x**2))**.5
    
    return n

def main():
    wavelength_nm_table = []
    water_refractive_idx = []

    min_refractive_index = 1000.0
    max_refractive_index = -1.0

    for wavelength_nm in range(360, 831, 1):
        n = calc_water_refraction(wavelength_nm)
        wavelength_nm_table.append(wavelength_nm)
        water_refractive_idx.append(n)

        min_refractive_index = min(min_refractive_index, n)
        max_refractive_index = max(max_refractive_index, n)

    # wavelength_nm_tableがx軸、water_refractive_idxがy軸のグラフを描画する
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(wavelength_nm_table, water_refractive_idx, color='#ff0000', lw=1.5)
    plt.title('Refractive Index of Water vs Wavelength')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Refractive Index')
    #plt.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', lw=0.8)

    plt.xlim(wavelength_nm_table[0], wavelength_nm_table[-1])
    plt.ylim(min_refractive_index, max_refractive_index)
    plt.show()

if __name__ == "__main__":
    main()
