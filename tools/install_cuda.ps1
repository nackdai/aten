# NOTE:
# How to invoke powershell script from command prompt.
# https://qiita.com/tomoko523/items/df8e384d32a377381ef9
# e.g. powershell -NoProfile -ExecutionPolicy Unrestricted .\install_cuda.ps1

# Option example:
# powershell -NoProfile -ExecutionPolicy Unrestricted .\install_cuda.ps1 -no_download

param (
    [switch]$no_download,   # Flag if script doesn't download packages.
    [switch]$no_copy        # Flag if script doesn't copy packages.
)

[string] $script:CUDA_VERSION = "12.5"
[string] $script:CUDA_PLATFORM = "windows-x86_64"
[string] $script:CUDA_PACKAGE_ARCHIVE_URL = "https://developer.download.nvidia.com/compute/cuda/redist"

[string] $script:CUDA_THRUST_VERSION = "1.15.0"
[string] $script:CUDA_THRUST_URL = "https://github.com/NVIDIA/thrust/archive/refs/tags"

[string] $script:CUDA_CUB_VERSION = "1.15.0"
[string] $script:CUDA_CUB_URL = "https://github.com/NVIDIA/cub/archive/refs/tags/"

[string] $script:CUDA_CXX_VERSION = "1.8.0-post-release"
[string] $script:CUDA_CXX_URL = "https://github.com/NVIDIA/libcudacxx/archive/refs/tags/"

[string] $script:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CUDA_VERSION"

[string] $script:CUDA_NVCC_PATCH_VERSION = "82"
[string] $script:CUDA_CUDART_PATCH_VERSION = "82"
[string] $script:CUDA_VS_INTEGRATION_PATCH_VERSION = "82"

[string] $script:ERROR_STORE_TXT = "stderr.txt"

function DownloadAndExtractPackage {
    param (
        [string]$package_archive,   # package archive.
        [string]$package_url        # URL to package archive.
    )

    # Download subpackages
    curl.exe -OL $package_url

    # Extract archive.
    7z x -y $package_archive 2>$ERROR_STORE_TXT
}

function DownloadCUDASubPackage {
    param (
        [string]$package,   # package name.
        [string]$version    # package version.
    )

    [string] $local:package_archive = "$package-$CUDA_PLATFORM-$version-archive.zip"
    [string] $local:package_url = "$CUDA_PACKAGE_ARCHIVE_URL/$package/$CUDA_PLATFORM/$package_archive"

    DownloadAndExtractPackage $package_archive $package_url
}

function DownloadPackage {
    param (
        [string]$package,   # package name.
        [string]$github_url # URL to github repo.
    )

    [string] $local:package_archive ="$package.zip"
    [string] $local:package_url = "$github_url/$package_archive"

    DownloadAndExtractPackage $package_archive $package_url
    if ($LastExitCode -ne 0) {
        Check7zipExtractError $ERROR_STORE_TXT
    }
}

function Check7zipExtractError {
    param (
        [string]$error_log_file
    )

    # Check content of error log text.
    [int] $local:exit_code = 0
    if (Test-Path -Path .\$error_log_file) {
        # If stderr.txt exist, there is a possibility 7zip raises an error.
        $exit_code = 1
        foreach($line in Get-Content .\$error_log_file) {
            # Remove unnecessary prefix.
            $line = $line.Replace("7z : ", "")

            # If the erorr mentions creating symbolic ling is faild, it's the ignorable error.
            if ($line -match "^(ERROR)") {
                if ($line -match "^(ERROR: Cannot create symbolic link).+$") {
                    $exit_code = 0
                } else {
                    Write-Host "Error log: [$line]" -f Red
                    exit 1
                }
            }
        }
    }
    else {
        Write-Host "No $error_log_file" -f Red
    }

    if ($exit_code -ne 0) {
        exit 1
    }
}

# Check if 7z is installed.
# https://stackoverflow.com/questions/11242368/test-if-executable-is-in-path-in-powershell
if ((Get-Command "7z" -ErrorAction SilentlyContinue) -eq $null) {
    Write-Host "7zip is not instanlled" -f Red
    exit 1
}

if (-not $no_download) {
    # nvcc
    Write-Host "Download cuda_nvcc" -f Green
    DownloadCUDASubPackage "cuda_nvcc" "$CUDA_VERSION.$CUDA_NVCC_PATCH_VERSION"
    if ($LastExitCode -ne 0) {
        exit 1
    }

    # cudart
    Write-Host "Download cuda_cudart" -f Green
    DownloadCUDASubPackage "cuda_cudart" "$CUDA_VERSION.$CUDA_CUDART_PATCH_VERSION"
    if ($LastExitCode -ne 0) {
        exit 1
    }

    # visual_studio_integration
    Write-Host "Download visual_studio_integration" -f Green
    DownloadCUDASubPackage "visual_studio_integration" "$CUDA_VERSION.$CUDA_VS_INTEGRATION_PATCH_VERSION"
    if ($LastExitCode -ne 0) {
        exit 1
    }

    # libcudacxx
    Write-Host "Download libcudacxx" -f Green
    DownloadPackage $CUDA_CXX_VERSION $CUDA_CXX_URL

    # cub
    Write-Host "Download cub" -f Green
    DownloadPackage $CUDA_CUB_VERSION $CUDA_CUB_URL

    # thrust
    Write-Host "Download thrust" -f Green
    DownloadPackage $CUDA_THRUST_VERSION $CUDA_THRUST_URL
}

function CreateDirectory {
    param (
        [string]$dir
    )

    Write-Host "Create $dir" -f Yellow

    if (!(Test-Path -Path $dir)) {
        New-Item -ItemType Directory -Path $dir
        Write-Host "$dir created successfully!" -f Green
    }
}

function CopyFilesInDirectory {
    param (
        [string]$src_dir,   # source directory.
        [string]$dst_dir    # destination directory.
    )

    Write-Host "Copy $src_dir\* to $dst_dir" -f Yellow

    Copy-Item -Path "$src_dir\*" -Destination "$dst_dir" -Recurse -Force
}

function CopyCUDASubPackage {
    param (
        [string]$package,   # package name.
        [string]$version,   # package version.
        [string]$dst_dir    # destination directory.
    )

    [string] $local:src_dir = "$package-$CUDA_PLATFORM-$version-archive"

    CopyFilesInDirectory $src_dir $dst_dir
}

if (-not $no_copy) {
    # Create directory.
    CreateDirectory $CUDA_PATH

    # nvcc
    Write-Host "Copy cuda_nvcc" -f Green
    CopyCUDASubPackage "cuda_nvcc" "$CUDA_VERSION.$CUDA_NVCC_PATCH_VERSION" $CUDA_PATH

    # cudart
    Write-Host "Copy cuda_cudart" -f Green
    CopyCUDASubPackage "cuda_cudart" "$CUDA_VERSION.$CUDA_CUDART_PATCH_VERSION" $CUDA_PATH

    # visual_studio_integration
    Write-Host "Copy visual_studio_integration" -f Green
    [string] $extra_dst_dir = "$CUDA_PATH\extras"
    CreateDirectory $extra_dst_dir
    CopyCUDASubPackage "visual_studio_integration" "$CUDA_VERSION.$CUDA_VS_INTEGRATION_PATCH_VERSION" $extra_dst_dir

    # libcudacxx
    Write-Host "Copy libcudacxx" -f Green
    [string] $cxx_src_dir = "libcudacxx-$CUDA_CXX_VERSION\include"
    [string] $cxx_dst_dir = "$CUDA_PATH\include"
    CreateDirectory $cxx_dst_dir
    CopyFilesInDirectory $cxx_src_dir $cxx_dst_dir

    # cub
    Write-Host "Copy cub" -f Green
    [string] $cub_src_dir = "cub-$CUDA_CUB_VERSION\cub"
    [string] $cub_dst_dir = "$CUDA_PATH\include\cub"
    CreateDirectory $cub_dst_dir
    CopyFilesInDirectory $cub_src_dir $cub_dst_dir

    # thrust
    Write-Host "Copy thrust" -f Green
    [string] $thrust_src_dir = "thrust-$CUDA_THRUST_VERSION\thrust"
    [string] $thrust_dst_dir = "$CUDA_PATH\include\thrust"
    CreateDirectory $thrust_dst_dir
    CopyFilesInDirectory $thrust_src_dir $thrust_dst_dir
}
