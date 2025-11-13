# Getting started

## Prerequisites

### Windows

Supported OS is `Windows 11`.

- To develop on Windows, install Visual Studio 2022.
- To develop for Linux on Windows, use Ubuntu on WSL2 (confirmed: Ubuntu 22.04 LTS).
  - Recommended editor is `Visual Studio Code` (can connect to the WSL2 Ubuntu instance).

#### CUDA

Verified CUDA version is `12.5`.

- If you change the CUDA version, select the desired CUDA in Visual Studio:
  1. Select the project in Visual Studio.
  2. Right-click → Build Dependencies → Build Customizations...
  3. Choose the CUDA version.

#### FBX SDK

`FbxConverter` depends on the [Autodesk FBX SDK](https://aps.autodesk.com/developer/overview/fbx-sdk)

- Configure in Visual Studio:
  1. Open Property Manager.
  2. Select the `FbxConverter` project.
  3. Under User `Macros`, set `FBXSDK` and `FBXSDK_LIB`.

#### Docker (WSL2)

To use Docker from Ubuntu on WSL2, install Docker Desktop on Windows.
The reference is <https://learn.microsoft.com/en-us/windows/wsl/tutorials/wsl-containers> .

### Linux

Supported distribution is `Ubuntu 22.04`. Any editor may be used. `Visual Studio Code` is
recommended. A devcontainer configuration is provided.

The development environment is Docker-based. Minimum setup references:

- [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
