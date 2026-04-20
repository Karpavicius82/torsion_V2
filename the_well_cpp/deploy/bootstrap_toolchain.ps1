# ============================================================================
# BOOTSTRAP_TOOLCHAIN.PS1 — Install bare-metal cross-compiler for Living Silicon
#
# Strategy (in priority order):
#   1. MSYS2 cross-gcc (pacman -S mingw-w64-cross-toolchain)
#   2. WSL2 Ubuntu gcc + nasm
#   3. Download prebuilt x86_64-elf-gcc from GitHub
#
# After install, verifies: g++ -ffreestanding -nostdlib compiles freestanding code
# ============================================================================

param(
    [string]$Method = "auto"  # auto | msys2 | wsl | download
)

$ErrorActionPreference = "Stop"

function Write-Silicon($msg) { Write-Host "[TOOLCHAIN] $msg" -ForegroundColor Cyan }
function Write-OK($msg)      { Write-Host "[  OK  ] $msg" -ForegroundColor Green }
function Write-ERR($msg)     { Write-Host "[ FAIL ] $msg" -ForegroundColor Red }

# ── Test source ──
$testSrc = @"
// Minimal bare-metal test
extern "C" {
    void _start() {
        volatile unsigned char* serial = (volatile unsigned char*)0x3F8;
        *serial = 'O';
        *serial = 'K';
        while(1) { asm volatile("hlt"); }
    }
}
"@

function Test-Compiler {
    param([string]$GCC, [string]$Label)

    Write-Silicon "Testing: $Label ($GCC)"
    $tmpSrc = [System.IO.Path]::GetTempFileName() + ".cpp"
    $tmpObj = [System.IO.Path]::ChangeExtension($tmpSrc, ".o")

    Set-Content -Path $tmpSrc -Value $testSrc -Encoding ASCII

    try {
        $result = & $GCC -ffreestanding -fno-exceptions -fno-rtti -nostdlib `
                        -mno-red-zone -mavx2 -O2 -std=c++20 `
                        -c $tmpSrc -o $tmpObj 2>&1

        if ((Test-Path $tmpObj) -and (Get-Item $tmpObj).Length -gt 0) {
            $size = (Get-Item $tmpObj).Length
            Write-OK "$Label works! Output: $size bytes"
            Remove-Item $tmpSrc, $tmpObj -ErrorAction SilentlyContinue
            return $true
        } else {
            Write-ERR "$Label failed: $result"
            Remove-Item $tmpSrc -ErrorAction SilentlyContinue
            return $false
        }
    } catch {
        Write-ERR "$Label error: $_"
        Remove-Item $tmpSrc -ErrorAction SilentlyContinue
        return $false
    }
}

# ============================================================================
# Strategy 1: MSYS2 cross-compiler
# ============================================================================
function Try-MSYS2 {
    Write-Silicon "Strategy 1: MSYS2 cross-compiler"

    if (-not (Test-Path "C:\msys64\usr\bin\bash.exe")) {
        Write-ERR "MSYS2 not found at C:\msys64"
        return $false
    }

    # Install cross-gcc if not present
    $crossGcc = "C:\msys64\opt\bin\x86_64-w64-mingw32-g++.exe"
    # Try different MSYS2 cross-compiler paths
    $candidates = @(
        "C:\msys64\ucrt64\bin\x86_64-w64-mingw32-g++.exe",
        "C:\msys64\mingw64\bin\x86_64-w64-mingw32-g++.exe",
        "C:\msys64\usr\bin\x86_64-pc-msys-g++.exe"
    )

    foreach ($gcc in $candidates) {
        if (Test-Path $gcc) {
            if (Test-Compiler -GCC $gcc -Label "MSYS2 ($gcc)") {
                Write-OK "USE: $gcc"
                return $true
            }
        }
    }

    # Try installing cross toolchain
    Write-Silicon "Installing MSYS2 cross-toolchain..."
    & C:\msys64\usr\bin\bash.exe -lc "pacman -S --noconfirm mingw-w64-cross-gcc 2>&1" | Out-Null

    foreach ($gcc in $candidates) {
        if (Test-Path $gcc) {
            if (Test-Compiler -GCC $gcc -Label "MSYS2 cross ($gcc)") {
                return $true
            }
        }
    }

    return $false
}

# ============================================================================
# Strategy 2: WSL2
# ============================================================================
function Try-WSL {
    Write-Silicon "Strategy 2: WSL2 Ubuntu"

    # Check WSL availability
    $distros = wsl --list --quiet 2>$null
    if (-not $distros) {
        Write-Silicon "No WSL distributions found. Installing Ubuntu..."
        wsl --install -d Ubuntu-24.04 --no-launch 2>$null
        Write-ERR "WSL Ubuntu installed. Please launch Ubuntu once to set up, then re-run."
        return $false
    }

    # Find a real Linux distro (not docker-desktop)
    $targetDistro = ""
    foreach ($line in ($distros -split "`n")) {
        $d = $line.Trim().Replace("`0", "")
        if ($d -and $d -ne "docker-desktop" -and $d -ne "docker-desktop-data") {
            $targetDistro = $d
            break
        }
    }

    if (-not $targetDistro) {
        Write-Silicon "No suitable WSL distro found. Installing Ubuntu..."
        wsl --install -d Ubuntu-24.04 2>$null
        Write-ERR "Ubuntu installed. Set up WSL, then re-run."
        return $false
    }

    Write-Silicon "Using WSL distro: $targetDistro"

    # Install gcc + nasm in WSL
    wsl -d $targetDistro -- bash -c "sudo apt-get update -qq && sudo apt-get install -y -qq g++ nasm binutils 2>&1" | Out-Null

    # Test freestanding compilation in WSL
    $wslTestSrc = "/tmp/silicon_test.cpp"
    $wslTestObj = "/tmp/silicon_test.o"

    wsl -d $targetDistro -- bash -c "cat > $wslTestSrc << 'EOFCPP'
extern ""C"" void _start() { volatile unsigned char* s = (volatile unsigned char*)0x3F8; *s='O'; while(1) __asm__(""hlt""); }
EOFCPP"

    $result = wsl -d $targetDistro -- bash -c "g++ -ffreestanding -fno-exceptions -fno-rtti -nostdlib -mno-red-zone -mavx2 -O2 -std=c++20 -c $wslTestSrc -o $wslTestObj 2>&1 && ls -la $wslTestObj && echo BUILD_OK || echo BUILD_FAIL"

    if ($result -match "BUILD_OK") {
        Write-OK "WSL2 cross-compilation works!"
        Write-Silicon "Build command: wsl -d $targetDistro -- bash -c 'cd /mnt/d/... && make kernel'"
        # Check nasm
        $nasmCheck = wsl -d $targetDistro -- bash -c "nasm --version 2>&1"
        Write-Silicon "NASM: $nasmCheck"
        return $true
    }

    Write-ERR "WSL2 compilation failed"
    return $false
}

# ============================================================================
# Strategy 3: Download prebuilt x86_64-elf-gcc
# ============================================================================
function Try-Download {
    Write-Silicon "Strategy 3: Download prebuilt x86_64-elf-gcc"

    $installDir = "C:\Silicon\toolchain"
    $gccPath = "$installDir\bin\x86_64-elf-g++.exe"

    if (Test-Path $gccPath) {
        if (Test-Compiler -GCC $gccPath -Label "Prebuilt x86_64-elf") {
            return $true
        }
    }

    # Download from known source
    $url = "https://github.com/lordmilko/i686-elf-tools/releases/latest/download/x86_64-elf-tools-windows.zip"
    $zipPath = "$env:TEMP\x86_64-elf-tools.zip"

    Write-Silicon "Downloading from $url ..."
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    } catch {
        Write-ERR "Download failed: $_"
        return $false
    }

    # Extract
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
    Expand-Archive -Path $zipPath -DestinationPath $installDir -Force
    Remove-Item $zipPath -ErrorAction SilentlyContinue

    if (Test-Path $gccPath) {
        if (Test-Compiler -GCC $gccPath -Label "Downloaded x86_64-elf") {
            # Add to PATH
            $env:PATH = "$installDir\bin;$env:PATH"
            Write-OK "Added to PATH: $installDir\bin"
            return $true
        }
    }

    Write-ERR "Download strategy failed"
    return $false
}

# ============================================================================
# MAIN
# ============================================================================
Write-Silicon "═══════════════════════════════════════════════════════"
Write-Silicon " Living Silicon — Toolchain Bootstrap"
Write-Silicon "═══════════════════════════════════════════════════════"

$success = $false

switch ($Method.ToLower()) {
    "auto" {
        $success = Try-MSYS2
        if (-not $success) { $success = Try-WSL }
        if (-not $success) { $success = Try-Download }
    }
    "msys2"    { $success = Try-MSYS2 }
    "wsl"      { $success = Try-WSL }
    "download" { $success = Try-Download }
}

if ($success) {
    Write-OK "Toolchain ready! You can now build Living Silicon."
    Write-Silicon "Next: .\build_image.ps1"
} else {
    Write-ERR "All strategies failed. Manual intervention required."
    Write-Silicon "Options:"
    Write-Silicon "  1. Install WSL2 Ubuntu: wsl --install -d Ubuntu-24.04"
    Write-Silicon "  2. Install Docker: docker pull dockcross/linux-x64"
    Write-Silicon "  3. Download x86_64-elf-gcc manually"
    exit 1
}
