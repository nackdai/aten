# NOTE:
# How to invoke powershell script from command prompt.
# https://qiita.com/tomoko523/items/df8e384d32a377381ef9
# e.g. powershell -NoProfile -ExecutionPolicy Unrestricted .\Build3rdParty.ps1

# Unfortunately, VsMSBuildCmd.bat can't work within PowerShell.
# It's hard to implement the same thing which VsMSBuildCmd.bat does in PowerShell.
# Therefore, calling Build3rdParty.bat directly.
# On the other hand, implementing switch flag to delete the directory is a bit messy in bat file.
# Therefore, this PowerShell script wrap the bat file and delete the directory.

param(
    [string]$config = "Debug",
    [string]$vs_edition = "Community",
    [switch]$clear
)

function DeleteDirectory {
    param (
        [string]$dir,
        [bool]$force_clear = $false
    )

    if ((Test-Path $dir) -And ($force_clear)) {
        Write-Host "Delete $dir" -f Yellow
        Remove-Item $dir -Force -Recurse
    }
}

DeleteDirectory "glfw\x64" $clear
DeleteDirectory "glew\build\vc16" $clear
DeleteDirectory "tinyobjloader\build" $clear
DeleteDirectory "assimp\build" $clear
DeleteDirectory "googletest\build" $clear

# NOTE:
# Why not use -Wait option. -Wait option takes a long time to return exit with using -PassThru option.
# https://www.d0web.com/blog/archives/2272
$proc = Start-Process ".\Build3rdParty.bat" -ArgumentList $config, $vs_edition -NoNewWindow -PassThru
$handle = $proc.Handle # cache proc.Handle
$proc.WaitForExit()
exit $proc.ExitCode
