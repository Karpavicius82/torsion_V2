# ============================================================================
# BUILD_IMAGE.PS1 — Build Living Silicon binary and bootable Hyper-V image
#
# Detects cross-compiler (WSL2 / MSYS2 / x86_64-elf), compiles kernel,
# assembles bootloader, links flat binary, creates bootable VHDX.
#
# Usage: .\build_image.ps1 [-Clean] [-SkipVHDX]
# ============================================================================

param(
    [switch]$Clean,
    [switch]$SkipVHDX,
    [string]$SrcDir = (Resolve-Path "$PSScriptRoot\..\src").Path,
    [string]$BuildDir = (Resolve-Path "$PSScriptRoot\..\build" -ErrorAction SilentlyContinue) ?? (Join-Path $PSScriptRoot "..\build"),
    [string]$VHDXPath = "C:\Silicon\well_silicon.vhdx"
)

$ErrorActionPreference = "Stop"

function Write-Silicon($msg) { Write-Host "[BUILD] $msg" -ForegroundColor Cyan }
function Write-OK($msg)      { Write-Host "[  OK ] $msg" -ForegroundColor Green }
function Write-ERR($msg)     { Write-Host "[ FAIL] $msg" -ForegroundColor Red }

# ============================================================================
# STEP 0: Ensure build directory
# ============================================================================
if (-not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null }
if ($Clean) {
    Remove-Item "$BuildDir\*" -Recurse -Force -ErrorAction SilentlyContinue
    Write-Silicon "Clean build"
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..") | Select-Object -ExpandProperty Path

# ============================================================================
# STEP 1: Detect compiler
# ============================================================================
Write-Silicon "Detecting cross-compiler..."

$BuildMethod = "none"
$WSLDistro = ""

# Check WSL distros
$distros = wsl --list --quiet 2>$null
foreach ($line in ($distros -split "`n")) {
    $d = $line.Trim().Replace("`0", "")
    if ($d -and $d -ne "docker-desktop" -and $d -ne "docker-desktop-data" -and $d.Length -gt 1) {
        # Test if g++ works
        $check = wsl -d $d -- bash -c "g++ --version 2>&1 | head -1" 2>$null
        if ($check -match "g\+\+") {
            $WSLDistro = $d
            $BuildMethod = "wsl"
            Write-OK "WSL2 distro: $WSLDistro ($check)"
            break
        }
    }
}

# Fallback: MSYS2
if ($BuildMethod -eq "none") {
    $msysGcc = "C:\msys64\ucrt64\bin\g++.exe"
    if (Test-Path $msysGcc) {
        $BuildMethod = "msys2"
        Write-OK "MSYS2 g++: $msysGcc"
    }
}

# Fallback: x86_64-elf in PATH
if ($BuildMethod -eq "none") {
    $elfGcc = Get-Command "x86_64-elf-g++" -ErrorAction SilentlyContinue
    if ($elfGcc) {
        $BuildMethod = "elf"
        Write-OK "x86_64-elf-g++: $($elfGcc.Source)"
    }
}

if ($BuildMethod -eq "none") {
    Write-ERR "No cross-compiler found. Run: .\bootstrap_toolchain.ps1"
    exit 1
}

# ============================================================================
# STEP 2: Compile kernel
# ============================================================================
Write-Silicon "Compiling kernel (bare-metal freestanding)..."

$CXXFLAGS = "-ffreestanding -fno-exceptions -fno-rtti -nostdlib -mno-red-zone -mavx2 -mfma -O3 -std=c++20 -DBARE_METAL"

switch ($BuildMethod) {
    "wsl" {
        # Convert Windows path to WSL mount path
        $wslProject = $ProjectRoot -replace '^([A-Z]):', { "/mnt/$($_.Groups[1].Value.ToLower())" } -replace '\\', '/'
        $wslSrc = "$wslProject/src"
        $wslBuild = "$wslProject/build"

        Write-Silicon "WSL path: $wslProject"

        # Compile kernel.o
        $result = wsl -d $WSLDistro -- bash -c "cd '$wslProject' && g++ $CXXFLAGS -I src -c src/kernel.cpp -o build/kernel.o 2>&1; echo EXIT=\$?"
        if ($result -match "EXIT=0") {
            Write-OK "kernel.o compiled"
        } else {
            Write-ERR "Compilation failed:"
            $result | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
            exit 1
        }

        # Assemble boot.o
        $nasmCheck = wsl -d $WSLDistro -- bash -c "which nasm 2>&1"
        if ($nasmCheck -match "nasm") {
            $result = wsl -d $WSLDistro -- bash -c "cd '$wslProject' && nasm -f elf64 src/boot/boot.asm -o build/boot.o 2>&1; echo EXIT=\$?"
            if ($result -match "EXIT=0") {
                Write-OK "boot.o assembled"
            } else {
                Write-ERR "NASM failed: $result"
                exit 1
            }
        } else {
            Write-Silicon "NASM not found — installing..."
            wsl -d $WSLDistro -- bash -c "sudo apt-get install -y -qq nasm 2>&1" | Out-Null
            $result = wsl -d $WSLDistro -- bash -c "cd '$wslProject' && nasm -f elf64 src/boot/boot.asm -o build/boot.o 2>&1; echo EXIT=\$?"
            if (-not ($result -match "EXIT=0")) {
                Write-ERR "NASM install/run failed"
                exit 1
            }
            Write-OK "boot.o assembled"
        }

        # Link
        $result = wsl -d $WSLDistro -- bash -c "cd '$wslProject' && ld -n -T src/boot/linker.ld -o build/well_silicon.bin build/boot.o build/kernel.o 2>&1; echo EXIT=\$?"
        if ($result -match "EXIT=0") {
            Write-OK "well_silicon.bin linked"
        } else {
            Write-ERR "Link failed: $result"
            exit 1
        }
    }

    "msys2" {
        # MSYS2 path
        $msysBash = "C:\msys64\usr\bin\bash.exe"
        $msysProject = $ProjectRoot -replace '\\', '/' -replace '^([A-Z]):',{ "/$($_.Groups[1].Value.ToLower())" }

        & $msysBash -c "cd '$msysProject' && /ucrt64/bin/g++ $CXXFLAGS -I src -c src/kernel.cpp -o build/kernel.o 2>&1; echo EXIT=\$?"
        # ... (similar pattern)
    }

    "elf" {
        & x86_64-elf-g++ $CXXFLAGS.Split(' ') -I $SrcDir -c "$SrcDir\kernel.cpp" -o "$BuildDir\kernel.o"
        & nasm -f elf64 "$SrcDir\boot\boot.asm" -o "$BuildDir\boot.o"
        & x86_64-elf-ld -n -T "$SrcDir\boot\linker.ld" -o "$BuildDir\well_silicon.bin" "$BuildDir\boot.o" "$BuildDir\kernel.o"
    }
}

# ============================================================================
# STEP 3: Verify binary
# ============================================================================
$binPath = Join-Path $BuildDir "well_silicon.bin"
if (Test-Path $binPath) {
    $size = (Get-Item $binPath).Length
    Write-OK "well_silicon.bin: $([math]::Round($size / 1024)) KB"
} else {
    Write-ERR "Binary not found at $binPath"
    exit 1
}

# ============================================================================
# STEP 4: Create bootable VHDX (optional)
# ============================================================================
if (-not $SkipVHDX) {
    Write-Silicon "Creating bootable VHDX..."

    $siliconDir = Split-Path $VHDXPath
    if (-not (Test-Path $siliconDir)) {
        New-Item -ItemType Directory -Path $siliconDir -Force | Out-Null
    }

    # Create VHDX if not exists
    if (-not (Test-Path $VHDXPath)) {
        New-VHD -Path $VHDXPath -SizeBytes 512MB -Dynamic | Out-Null
        Write-OK "Created VHDX: $VHDXPath"

        # Initialize disk
        Mount-VHD -Path $VHDXPath
        $disk = Get-VHD -Path $VHDXPath
        Initialize-Disk -Number $disk.DiskNumber -PartitionStyle MBR
        $part = New-Partition -DiskNumber $disk.DiskNumber -UseMaximumSize -AssignDriveLetter
        Format-Volume -Partition $part -FileSystem FAT32 -NewFileSystemLabel "SILICON" -Confirm:$false
        $letter = $part.DriveLetter
    } else {
        Mount-VHD -Path $VHDXPath
        $disk = Get-VHD -Path $VHDXPath
        $part = Get-Partition -DiskNumber $disk.DiskNumber | Where-Object { $_.Type -ne 'Reserved' } | Select-Object -First 1
        $letter = $part.DriveLetter
        if (-not $letter) {
            Add-PartitionAccessPath -DiskNumber $disk.DiskNumber -PartitionNumber $part.PartitionNumber -AssignDriveLetter
            $part = Get-Partition -DiskNumber $disk.DiskNumber | Where-Object { $_.DriveLetter } | Select-Object -First 1
            $letter = $part.DriveLetter
        }
    }

    # Copy kernel
    Copy-Item -Path $binPath -Destination "${letter}:\well_silicon.bin" -Force
    Write-OK "Kernel installed to ${letter}:\well_silicon.bin"

    Dismount-VHD -Path $VHDXPath
    Write-OK "VHDX ready: $VHDXPath"
}

# ============================================================================
# SUMMARY
# ============================================================================
Write-Silicon ""
Write-Silicon "═══════════════════════════════════════════════════════"
Write-Silicon " BUILD COMPLETE"
Write-Silicon "═══════════════════════════════════════════════════════"
Write-Silicon " Binary:  $binPath ($([math]::Round((Get-Item $binPath).Length / 1024)) KB)"
if (-not $SkipVHDX) { Write-Silicon " Image:   $VHDXPath" }
Write-Silicon " Method:  $BuildMethod"
Write-Silicon ""
Write-Silicon " NEXT STEPS:"
Write-Silicon "   .\hyperv_setup.ps1 -Action setup -KernelBin $binPath"
Write-Silicon "   .\hyperv_setup.ps1 -Action start"
Write-Silicon "   .\silicon_controller.exe"
