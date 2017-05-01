# aten

This is easy, simple path tracer.

## features


- Rendering algorithms
  - Ray Tracing
  - Next Event Estimation Path Tracing
  - Bi-Directional Path Tracing
  - Primary Sample Space Metropolis Light Transport
  - Energy Distribution Path Tracing
  - AOV
    - albedo, depth, normal, geometry id.
- Materials
  - Emissive
  - Lambert
  - Specular
  - Refraction
  - Microfacet Blinn
  - Microfacet Beckman
  - Microfacet GGX
  - Disney BRDF
  - Toon(Non Photoreal)
  - Layer
- Lights
  - Polygonal Light(Area Light)
  - Point Light
  - Spot Light
  - Directional Light
  - Image Based Lighting
- Quasi Monte Carlo
  - Halton
  - Sobol
- Rendering shapes
  - Polygon(.obj file)
  - Sphere
  - Cube
- Instancing
- Texture Map
  - Albedo
  - Normal
  - Roughness
- Denoise filter
  - Non Local Mean(CPU/GPU)
  - Birateral(CPU/GPU)
  - Practical Noise Reduction for Progressive Stochastic Ray Tracing with Perceptual Control(CPU)
  - Robust Image Denoising using a Virtual Flash Image for Monte Carlo Ray Tracing(CPU)
- PostEffect
  - Simple Bloom
  - Reinherd Tone Mapping
- Scene Definition by XML

## Limitations
  
- Not optimisation by SIMD
  - This should be easy, simple, to avoid difficult to understand, so not use SIMD.
- Bi-Directional Path Tracing
  - Only Area Light
- Scene Definition by XML
  - Not specify all definitions.
- Only for Windows.

## T.B.D

- GPGPU
  - idaten is GPGPU project.
- More efficient denoise filter
- Specify all scene definitions by XML
- Bi-Directional Path Tracing
  - Use all lights.

## Gallery

PathTracing 100spp
![PathTracing](gallery/pt100.png)

Materials PathTracing 100spp
![Materials](gallery/pt100_mtrl.png)

Bi-Directional PathTracing 100spp
![BDPT](gallery/bdpt100.png)

PSSMLT 10spp 100mlt 100mutation
![PSSMLT](gallery/pssmlt_10spp_100mutation_100mlt.png)

ERPT 10spp 10mutation
![ERPT](gallery/erpt_10spp_10mutation.png)