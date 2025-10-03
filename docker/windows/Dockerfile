# NOTE:
# For windows, --platform options is necessary.
#   docker build --platform windows -t <tag> .

# NOTE:
# To run the container with cmd.exe.
#   docker run -it --rm --platform windows -v "$(pwd):C:\work" mcr.microsoft.com/windows:20H2 cmd.exe
# The docker image built from this Dockerfile specifies powershell as entrypoint.
# Therefore, if it's fine to launch with powershell, we don't need to specify any executable like the following:
#   docker run -it --rm --platform windows -v "$(pwd):C:\work" <docker image name>:<tag>

FROM mcr.microsoft.com/windows/servercore:20H2

# Following 4 commands are for installing Chocolatey.
RUN powershell -Command "Invoke-WebRequest -Uri 'https://download.visualstudio.microsoft.com/download/pr/2d6bb6b2-226a-4baa-bdec-798822606ff1/8494001c276a4b96804cde7829c04d7f/ndp48-x86-x64-allos-enu.exe' -OutFile 'C:\ndp48.exe'"
RUN C:\ndp48.exe /q /norestart /log C:\ndp48.log && del C:\ndp48.exe

RUN powershell -Command "New-Item -Path 'C:\ProgramData' -ItemType Directory -Force"
RUN powershell -Command "New-Item -Path 'C:\Users\ContainerAdministrator\AppData\Local\Temp' -ItemType Directory -Force"

# Install Chocolatey.
# https://zenn.dev/kazuma_r5/articles/a6d2608446ebdf
RUN @"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" \
    -NoProfile \
    -InputFormat None \
    -ExecutionPolicy Bypass \
    -Command " [System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"

RUN @"C:\ProgramData\chocolatey\bin\choco.exe" install git.install -y

RUN @"C:\ProgramData\chocolatey\bin\choco.exe" \
    install cmake.install -y --installargs '"ADD_CMAKE_TO_PATH=User"'

RUN @"C:\ProgramData\chocolatey\bin\choco.exe" install 7zip.install -y

# Install visual studio 2022 community with only minimum components.
RUN @"C:\ProgramData\chocolatey\bin\choco.exe" \
    install visualstudio2022community \
    --package-parameters \
    "--add Microsoft.VisualStudio.Workload.VCTools \
    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
    --add Microsoft.VisualStudio.Component.Windows10SDK.19041" -y

# NOTE:
# If we'd like to install cuda via Chocolatey.
# https://community.chocolatey.org/packages/cuda#versionhistory
# RUN choco install cuda --version=<version> -y

ENTRYPOINT [ "powershell" ]
