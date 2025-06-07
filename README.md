<!-- markdownlint-disable MD024 MD029 MD033 -->
# aten

![CI](https://github.com/nackdai/aten/workflows/CI/badge.svg)

This is an easy, simple ray tracing renderer.

Aten is the Egyptian sun god.

Idaten (path tracing on GPGPU) is under construction.

Idaten is a Japanese god who runs fast.
"Idaten" includes the characters of aten: "id**aten**".

## Features

- Rendering algorithms
  - Next Event Estimation Path Tracing
- Acceleration
  - BVH
  - SBVH
  - TopLayer, BottomLayer
  - Transform (Translate, Rotate, Scale) objects
- Materials
  - Emissive
  - Lambert
  - Specular
  - Refraction
  - Microfacet Beckmann
  - Microfacet GGX
  - Oren-Nayar
  - Disney BRDF
  - Retroreflective (Experimental)
- Lights
  - Polygonal Light (Area Light)
  - Point Light
  - Spot Light
  - Directional Light
  - Image Based Lighting
- Quasi Monte Carlo
  - CMJ (Correlated Multi Jittered)
- Rendering shapes
  - Polygon (.obj file)
  - Sphere
- Texture Map
  - Albedo
  - Normal
  - Roughness
- Post Effect
  - Reinhard Tone Mapping
- Camera
  - Pinhole
  - Equirectangular
- Volume Rendering
  - Homogeneous
  - Heterogeneous
  - NanoVDB format
- Others
  - Instancing
  - Deformation
  - Alpha blending
- [Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination](https://cg.ivd.kit.edu/svgf.php)
- [Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting (ReSTIR)](https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf)
- [Physically-based Feature Line Rendering](http://lines.rexwe.st/)

## Limitations

- Not optimized with SIMD
  - To keep things easy and simple, SIMD is not used.
- Some comments are still in Japanese.

## How to Build

[How to Build](docs/how_to_build.md)

## How to Run

[How to Run](docs/how_to_run.md)

## Gallery

Path Tracing 100 spp

![PathTracing](docs/gallery/pt100.png)

Materials Path Tracing 100 spp

![Materials](docs/gallery/pt100_mtrl.png)

SVGF (1 spp / 5 bounces)

![SVGF_sponza](docs/gallery/svgf_1spp_sponza.png)
![SVGF_cryteksponza](docs/gallery/svgf_1spp_cryteksponza.png)

Deformation

![Deformation](docs/gallery/deform.png)

(c) Unity Technologies Japan/UCL

ReSTIR (1 spp / 5 bounces / 126 point lights, without environment map)

![ReSTIR](docs/gallery/compare_restir.png)

Alpha Blending

![AlphaBlending](docs/gallery/alpha_blend.png)

Physically-based Feature Line Rendering

![FeatureLine](docs/gallery/feature_line.png)

Homogeneous medium

![Homogeneous](docs/gallery/homogeneous.png)

Heterogeneous medium

![Heterogeneous](docs/gallery/heterogeneous.png)

## Development

- [Docker](docker/README.md)
- [Tools](tools/README.md)
- [Lint and Format](docs/lint_and_format.md)
- [Python Development](docs/python_development.md)
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)

### For VSCode development

We can open this project in a VSCode devcontainer.
If we encounter a devcontainer build failure, it might be due to the docker-compose version.
In that case, please update docker-compose.
