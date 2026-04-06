import math
import matplotlib.pyplot as plt

def calc_marshall_palmer_dropletsize_distribution_as_cm(droplet_diameter_cm: float, rainfall_rate_mm_hr: float) -> float:
    """Calculate the droplet size distribution of raindrops using the Marshall-Palmer distribution.

    The Marshall-Palmer distribution is given by:
    N(D) = N0 * exp(-Lambda * D)
    where:
    - N(D) is the number of droplets per unit volume per unit diameter interval
    - N0 is the intercept parameter (typically around 8000 m^-3 mm^-1)
    - Lambda is the slope parameter (related to the rainfall rate)
    - D is the diameter of the droplets

    Args:
        droplet_diameter_cm (float): The diameter of the raindrop in centimeters.
        rainfall_rate_mm_hr (float): The rainfall rate in millimeters per hour.

    Returns:
        float: The droplet size distribution for a given diameter D.
    """
    # Parameters for the Marshall-Palmer distribution
    N0 = 0.008  # cm^-4
    Lambda = 41 * (rainfall_rate_mm_hr ** -0.21)  # cm^-1

    # Calculate the droplet size distribution
    N_D = N0 * math.exp(-Lambda * droplet_diameter_cm)

    return N_D

def calc_marshall_palmer_dropletsize_distribution_as_mm(droplet_diameter_mm: float, rainfall_rate_mm_hr: float) -> float:
    """Calculate the droplet size distribution of raindrops using the Marshall-Palmer distribution.

    The Marshall-Palmer distribution is given by:
    N(D) = N0 * exp(-Lambda * D)
    where:
    - N(D) is the number of droplets per unit volume per unit diameter interval
    - N0 is the intercept parameter (typically around 8000 m^-3 mm^-1)
    - Lambda is the slope parameter (related to the rainfall rate)
    - D is the diameter of the droplets

    Args:
        droplet_diameter_mm (float): The diameter of the raindrop in millimeters.
        rainfall_rate_mm_hr (float): The rainfall rate in millimeters per hour.

    Returns:
        float: The droplet size distribution for a given diameter D.
    """
    # Parameters for the Marshall-Palmer distribution
    N0 = 8000  # m^-1 mm^-1
    Lambda = 4.1 * (rainfall_rate_mm_hr ** -0.21)  # mm^-1

    # Calculate the droplet size distribution
    N_D = N0 * math.exp(-Lambda * droplet_diameter_mm)

    return N_D

results: list[float] = []

# Rainfall rates in mm/hr
R = [1, 5, 25]

# 0 cm to 0.5 cm in 0.01 cm increments
D_cm = [x * 0.01 for x in range(0, 51)]

# 0 mm to 5 mm in 0.1 mm increments
D_mm = [x * 0.1 for x in range(0, 51)]

for rainfall_rate in R:
    results.append(
        [calc_marshall_palmer_dropletsize_distribution_as_mm(d, rainfall_rate) for d in D_mm]
    )

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(D_mm, results[0], color='#ff0000', lw=1.5, label='R = 1 mm/hr')
plt.plot(D_mm, results[1], color='#00ff00', lw=1.5, label='R = 5 mm/hr')
plt.plot(D_mm, results[2], color='#0000ff', lw=1.5, label='R = 25 mm/hr')
plt.xlabel('Droplet Diameter (mm)')
plt.ylabel('Droplet Size Distribution')
plt.title('Marshall-Palmer Droplet Size Distribution')
plt.yscale('log', base=10)

# For mm, [10e-2, 10e3]
# For cm, [10e-6, 10e-1] 
plt.ylim(10e-2, 10e3)

plt.legend()
plt.show()