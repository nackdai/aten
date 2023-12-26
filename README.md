<!-- markdownlint-disable MD024 MD029 MD033 -->
# aten

![CI](https://github.com/nackdai/aten/workflows/CI/badge.svg)

This is easy, simple ray tracing renderer.

Aten is Egyptian sun god.

Idaten(path tracing on GPGPU) is under construction.

Idaten is Japanese god, it runs fast.
And Idanten includes characters of aten, "id**aten**"

## Features

**Some features are only supported by either.**

- Rendering algorithms
  - Next Event Estimation Path Tracing
- Acceralation
  - BVH
  - SBVH
  - TopLayer, BottomLayer
  - Transform(Translate, Rotate, Scale) objects
- Materials
  - Emissive
  - Lambert
  - Specular
  - Refraction
  - Microfacet Blinn
  - Microfacet Beckman
  - Microfacet GGX
  - OrenNayar
  - Disney BRDF
  - CarPaint (Experimental)
  - Retroreflective (Experimental)
- Lights
  - Polygonal Light(Area Light)
  - Point Light
  - Spot Light
  - Directional Light
  - Image Based Lighting
- Quasi Monte Carlo
  - Halton
  - Sobol
  - CMJ(Correllated Multi Jittered)
- Rendering shapes
  - Polygon(.obj file)
  - Sphere
- Texture Map
  - Albedo
  - Normal
  - Roughness
- Denoise filter
  - Non Local Mean
- PostEffect
  - Simple Bloom
  - Reinherd Tone Mapping
- Camera
  - Pinhole
  - 360 view
- Others
  - Instancing
  - Deformation
  - Alpha blending
- [Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination](https://cg.ivd.kit.edu/svgf.php)
- [Spatiotemporal reservoir resampling for real-time ray tracing
with dynamic direct lighting](https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf)
- [Physically-based Feature Line Rendering](http://lines.rexwe.st/)

## Limitations

- Not optimisation by SIMD
  - This should be easy, simple, to avoid difficult to understand, so not use SIMD.
- There are some Japanese comments...

## How To Build

See [How To Build](docs/how_to_build.md) document

## How to run

### Windows

Please find `exe` files and run them. You can find them in each directories
where source files are in.

### Linux

Please find execution files and run them. You can find them in the directories
in the directory which you built the applications. And the directories have
same name as execution file.

### <a name="RunOnDocker">Docker</a>

This section works for ony Linux.

If you would like to run built applications in docker, you need to ensure that your host can accept
X forwarded connections:

```shell
xhost +local:<Docker container name>`
```

And, run docker container like the following:

```shell
docker run -it --rm -v ${PWD}:/work -v /tmp/.X11-unix:/tmp/.X11-unix:rw --runtime=nvidia -e DISPLAY <Image Name>:latest bash
```

#### docker-compose

You also need to ensure your host accept X forwared connections.
See [Docker in How to run](#RunOnDocker)

And, run docker container via docker-compose like the following:

```shell
docker-compose -f .devcontainer/docker-compose.yml run aten
```

## For VSCode development

You can open this project on VSCode devcontainer.
If you face on devcontainer build failure, it might be due to docker-compose version. In that case,
please update docker-compose.s

## Gallery

PathTracing 100spp

![PathTracing](docs/gallery/pt100.png)

Materials PathTracing 100spp

![Materials](docs/gallery/pt100_mtrl.png)

SVGF (1spp/5bounds)

![SVGF_sponza](docs/gallery/svgf_1spp_sponza.png)
![SVGF_cryteksponza](docs/gallery/svgf_1spp_cryteksponza.png)

Deformation

![Deformation](docs/gallery/deform.png)

(c) Unity Technologies Japan/UCL

ReSTIR (1spp/5bounds/126point lights w/o environment map)

![ReSTIR](docs/gallery/compare_restir.png)

CarPaint

![CarPaint](docs/gallery/car_paint.png)

Retroreflective

![Retroreflective](docs/gallery/retroreflective.gif)

AlphaBlending

![AlphaBlending](docs/gallery/alpha_blend.png)

Physically-based Feature Line Rendering

![FeatureLine](docs/gallery/feature_line.png)
