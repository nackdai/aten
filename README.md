# aten

![](https://github.com/nackdai/aten/workflows/.github/workflows/CI/badge.svg)

[![CircleCI](https://circleci.com/gh/nackdai/aten.svg?style=svg)](https://circleci.com/gh/nackdai/aten)

This is easy, simple ray tracing renderer.

Aten is Egyptian sun god.

Idaten(path tracing on GPGPU) is under construction.

Idaten is Japanese god, it runs fast.
And Idanten includes characters of aten, "id**aten**"

## Features

- Rendering algorithms
  - Ray Tracing
  - Next Event Estimation Path Tracing
  - Bi-Directional Path Tracing
  - Primary Sample Space Metropolis Light Transport
  - Energy Re-distribution Path Tracing
  - AOV
    - albedo, depth, normal, geometry id.
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
  - Disney BRDF
  - Car Paint (Experimental)
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
  - CMJ(Correllated Multi Jittered)
- Rendering shapes
  - Polygon(.obj file)
  - Sphere
  - Cube
- Texture Map
  - Albedo
  - Normal
  - Roughness
- Denoise filter
  - Non Local Mean
  - Birateral
  - Practical Noise Reduction for Progressive Stochastic Ray Tracing with Perceptual Control
  - Robust Image Denoising using a Virtual Flash Image for Monte Carlo Ray Tracing
  - [Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination](https://cg.ivd.kit.edu/svgf.php)
- PostEffect
  - Simple Bloom
  - Reinherd Tone Mapping
- Cameras
  - Pinhole
  - Thin Lens(DoF)
  - 360 view
- Others
  - Instancing
  - Scene Definition by XML
  - Deformation

## Limitations
  
- Not optimisation by SIMD
  - This should be easy, simple, to avoid difficult to understand, so not use SIMD.
- Bi-Directional Path Tracing
  - Only Area Light
- Scene Definition by XML
  - Can not specify all definitions.
- Not support texture alpha yet.
- There are some Japanese comments...

## Under Construction Works

- More efficient denoise filter
  - [Gradient Estimation for Real-Time Adaptive Temporal Filtering](https://cg.ivd.kit.edu/atf.php)

## Future Works

- Specify all scene definitions by XML
- Bi-Directional Path Tracing
  - Enable to use all lights
- Sub Surface Scattering
- Particiate Media
- Rendering algorithms
  - Photon mapping
  - Progressive Photom mapping
  - VCM
- More efficient acceleration algorithms

## How To Build

### Prepearing

You also should get submodules in `3rdparty` directory.
To do that, follow as below:

```shell
git submodule update --init --recursive
```

### Build on Windows

1. Install `CUDA 10.1` and depended NVIDIA driver
2. Run `aten/3rdparty/Build3rdParty.bat <Debug|Release>`
3. Launch `aten/vs2019/aten.sln`
4. Build porjects with `x64` (not support `x86`)

I confirmed with Visual Studio 2019 on Windows10.

Supoort just only `CUDA 10.1`.

### Build on Linux

1. Install `CUDA 10.1` or later and depended NVIDIA driver
1. Install applications (You can find what you need in `Dockerfile`)
    1. Install `cmake` `3.10.0` or later
    1. Install `clang 8.0.0`
    1. Install `ninja-build`
1. `cd aten/build`
1. `./RunCMake.sh <Build Type> <Compute Capability>`
1. Run make `ninja`

I confirmed on Ubuntu16.04 LTS and 18.04 LTS.

#### What is RunCMake.sh

`RunCMake.sh` is a scirpt to help you to build `aten` with CMake.
It is located in `scripts` directory. If you would like to use it.
Copy it to the build directory you want.

It needs 2 arguments as below:

1. Build Type: `Debug` or `Release`
1. Compute Capability: It depends on your GPU. But, you need to specify it
without `.`. For example, if `Comnpute Capability` is `7.5`, please specify
like `75`.

Example to run `RunCMake.sh` is below:

```shell
./RunCMake.sh Release 75
```

You can get `Comnpute Capability` with running `get_cuda_sm.sh`.
If you don't specify `Comnpute Capability`, while configuring `CMakeLists.txt`,
`get_cuda_sm.sh` run and `Comnpute Capability` is specified.


#### Build on Docker

You can build and run executable aten application on Docker container.

##### Build on only Docker

1. Install `Docker`
2. Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).
3. Move to `aten` directory
4. Build docker image like below:

```shell
docker build -t <Any name> .
```

5. Run docker container like below:

```shell
docker run -it --rm -v ${PWD}:/work -v /tmp/.X11-unix:/tmp/.X11-unix:rw --runtime=nvidia -e DISPLAY <Image name>:latest bash
```

6. `mkdir aten/build`
7. `cd aten/build`
8. `cp ../scripts/RunCMake.sh .`
9. `./RunCMake.sh <Build Type> <Compute Capability>`
10. Run make `ninja`

##### Build with docker-compose

1. Install `Docker` and `docker-compose`
1. Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker).
1. Build docker image `docker-compose build`
1. Run docker container `docker-compose run aten`
1. `mkdir aten/build`
1. `cd aten/build`
1. `cp ../scripts/RunCMake.sh .`
1. `./RunCMake.sh <Build Type> <Compute Capability>`
1. Run make `ninja`
 
## How to run

### Run on Windows

Please find `exe` files and run them. You can find them in each directories
where source files are in.

### Run on Linux

Please find execution files and run them. You can find them in the directories
in the directory which you built the applications. And the directories have
same name as execution file.

### Run on Docker

If you would like to run built applications in docker, you need to ensure that
your host can accept X forwarded connections
`xhost +local:<Docker container name>`.

And, run docker container like below:

```shell
docker run -it --rm -v ${PWD}:/work -v /tmp/.X11-unix:/tmp/.X11-unix:rw --runtime=nvidia -e DISPLAY <Image Name>:latest bash
```

### Run with docker-compose

You alse need to ensure your host as [Run on Docker](#Run-on-Docker).

And, run docker container via docker-compose like below:

```shell
docker-compose run aten
```

## Gallery

PathTracing 100spp

![PathTracing](gallery/pt100.png)

Materials PathTracing 100spp

![Materials](gallery/pt100_mtrl.png)

Bi-Directional PathTracing 100spp

![BDPT](gallery/bdpt100.png)

PSSMLT 10spp 10mlt 10mutation

![PSSMLT](gallery/pssmlt_10spp_10mutation_10mlt.png)

GPGPU 1spp

![GPGPU_1spp](gallery/gpgpu_1spp.png)

Deformation

![Deformation](gallery/deform.png)

(c) Unity Technologies Japan/UCL
