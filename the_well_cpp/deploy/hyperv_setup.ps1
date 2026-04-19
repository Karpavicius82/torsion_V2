#Requires -RunAsAdministrator
# ============================================================================
# LIVING SILICON — Hyper-V Deployment Module
# Industrial-grade VM setup for bare-metal physics kernel
# ============================================================================

param(
    [string]$Action = "setup",           # setup | start | stop | restart | destroy | status
    [string]$VMName = "LivingSilicon",
    [int]$CPUCount = 6,
    [int64]$MemoryMB = 2048,
    [string]$SiliconDir = "C:\Silicon",
    [string]$KernelBin = "",             # path to well_silicon.bin
    [string]$SerialPipe = "\\.\pipe\SiliconSerial"
)

$ErrorActionPreference = "Stop"

# ── Colors ──
function Write-Silicon($msg) { Write-Host "[SILICON] $msg" -ForegroundColor Cyan }
function Write-OK($msg)      { Write-Host "[  OK  ] $msg" -ForegroundColor Green }
function Write-ERR($msg)     { Write-Host "[ FAIL ] $msg" -ForegroundColor Red }

# ============================================================================
# PHASE 1: Ensure Hyper-V is enabled
# ============================================================================
function Ensure-HyperV {
    Write-Silicon "Checking Hyper-V status..."
    $feature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V
    if ($feature.State -ne "Enabled") {
        Write-Silicon "Enabling Hyper-V (requires reboot)..."
        Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All -NoRestart
        Write-ERR "Hyper-V enabled. REBOOT REQUIRED. Then re-run this script."
        exit 1
    }
    Write-OK "Hyper-V is enabled"
}

# ============================================================================
# PHASE 2: Create infrastructure
# ============================================================================
function Setup-Infrastructure {
    # Create directory
    if (-not (Test-Path $SiliconDir)) {
        New-Item -ItemType Directory -Path $SiliconDir -Force | Out-Null
        Write-OK "Created $SiliconDir"
    }

    # Create private VMSwitch (isolated, no internet)
    $switchName = "SiliconSwitch"
    $existing = Get-VMSwitch -Name $switchName -ErrorAction SilentlyContinue
    if (-not $existing) {
        New-VMSwitch -SwitchName $switchName -SwitchType Private | Out-Null
        Write-OK "Created private VMSwitch: $switchName"
    } else {
        Write-OK "VMSwitch '$switchName' already exists"
    }
}

# ============================================================================
# PHASE 3: Create VM
# ============================================================================
function Create-SiliconVM {
    $existing = Get-VM -Name $VMName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-OK "VM '$VMName' already exists (State: $($existing.State))"
        return
    }

    Write-Silicon "Creating VM: $VMName [$CPUCount cores, ${MemoryMB}MB RAM]"

    # Create VHDX
    $vhdxPath = Join-Path $SiliconDir "well_silicon.vhdx"
    if (-not (Test-Path $vhdxPath)) {
        New-VHD -Path $vhdxPath -SizeBytes 512MB -Dynamic | Out-Null
        Write-OK "Created VHDX: $vhdxPath"
    }

    # Create VM (Generation 1 for BIOS/Multiboot compatibility)
    New-VM -Name $VMName `
           -MemoryStartupBytes ($MemoryMB * 1MB) `
           -Generation 1 `
           -SwitchName "SiliconSwitch" `
           -VHDPath $vhdxPath | Out-Null

    # Configure CPU
    Set-VMProcessor -VMName $VMName `
                    -Count $CPUCount `
                    -Reserve 100 `
                    -RelativeWeight 200 `
                    -CompatibilityForMigrationEnabled $false `
                    -ExposeVirtualizationExtensions $false

    # Configure Memory (no dynamic memory — fixed allocation)
    Set-VMMemory -VMName $VMName `
                 -DynamicMemoryEnabled $false `
                 -StartupBytes ($MemoryMB * 1MB)

    # Serial port → Named Pipe (host communication)
    Set-VMComPort -VMName $VMName `
                  -Number 1 `
                  -Path $SerialPipe

    # Disable checkpoints (no snapshots — bare-metal doesn't need them)
    Set-VM -VMName $VMName `
           -CheckpointType Disabled `
           -AutomaticStartAction Nothing `
           -AutomaticStopAction TurnOff `
           -Notes "Living Silicon OS — bare-metal physics kernel"

    # Disable integration services (bare-metal kernel doesn't support them)
    Get-VMIntegrationService -VMName $VMName | ForEach-Object {
        Disable-VMIntegrationService -VMName $VMName -Name $_.Name -ErrorAction SilentlyContinue
    }

    Write-OK "VM '$VMName' created successfully"
    Write-Silicon "  CPU:    $CPUCount cores (100% reserved)"
    Write-Silicon "  RAM:    ${MemoryMB}MB (fixed)"
    Write-Silicon "  Serial: $SerialPipe"
    Write-Silicon "  VHDX:   $vhdxPath"
}

# ============================================================================
# PHASE 4: Install kernel to VHDX
# ============================================================================
function Install-Kernel {
    param([string]$BinPath)

    if (-not $BinPath -or -not (Test-Path $BinPath)) {
        Write-ERR "Kernel binary not found: $BinPath"
        Write-Silicon "Usage: .\hyperv_setup.ps1 -Action setup -KernelBin path\to\well_silicon.bin"
        return
    }

    $vhdxPath = Join-Path $SiliconDir "well_silicon.vhdx"
    Write-Silicon "Installing kernel to VHDX..."

    # Mount VHDX
    Mount-VHD -Path $vhdxPath
    $disk = Get-VHD -Path $vhdxPath
    $diskNum = $disk.DiskNumber

    # Initialize disk if needed
    $part = Get-Partition -DiskNumber $diskNum -ErrorAction SilentlyContinue
    if (-not $part) {
        Initialize-Disk -Number $diskNum -PartitionStyle MBR
        $part = New-Partition -DiskNumber $diskNum -UseMaximumSize -AssignDriveLetter
        Format-Volume -Partition $part -FileSystem FAT32 -NewFileSystemLabel "SILICON" -Confirm:$false
    }

    $driveLetter = ($part | Where-Object { $_.Type -ne 'Reserved' } | Select-Object -First 1).DriveLetter
    if (-not $driveLetter) {
        $part | Add-PartitionAccessPath -AssignDriveLetter
        $driveLetter = ($part | Get-Partition | Where-Object { $_.DriveLetter }).DriveLetter
    }

    # Copy kernel binary
    $dest = "${driveLetter}:\well_silicon.bin"
    Copy-Item -Path $BinPath -Destination $dest -Force
    Write-OK "Kernel installed: $dest ($(((Get-Item $dest).Length / 1KB).ToString('N1')) KB)"

    # Dismount
    Dismount-VHD -Path $vhdxPath
    Write-OK "VHDX dismounted"
}

# ============================================================================
# PHASE 5: VM Control
# ============================================================================
function Start-Silicon {
    $vm = Get-VM -Name $VMName -ErrorAction SilentlyContinue
    if (-not $vm) { Write-ERR "VM '$VMName' not found"; return }
    if ($vm.State -eq 'Running') { Write-OK "VM already running"; return }
    Start-VM -Name $VMName
    Write-OK "VM '$VMName' started — serial output on: $SerialPipe"
    Write-Silicon "Connect: .\silicon_controller.exe"
}

function Stop-Silicon {
    $vm = Get-VM -Name $VMName -ErrorAction SilentlyContinue
    if (-not $vm) { Write-ERR "VM '$VMName' not found"; return }
    if ($vm.State -eq 'Off') { Write-OK "VM already off"; return }
    Stop-VM -Name $VMName -Force -TurnOff
    Write-OK "VM '$VMName' stopped"
}

function Restart-Silicon {
    Stop-Silicon
    Start-Sleep -Seconds 2
    Start-Silicon
}

function Get-SiliconStatus {
    $vm = Get-VM -Name $VMName -ErrorAction SilentlyContinue
    if (-not $vm) { Write-ERR "VM '$VMName' not found"; return }

    Write-Silicon "═══════════════════════════════════════"
    Write-Silicon " VM:     $VMName"
    Write-Silicon " State:  $($vm.State)"
    Write-Silicon " CPU:    $($vm.ProcessorCount) cores"
    Write-Silicon " RAM:    $([math]::Round($vm.MemoryAssigned / 1MB))MB"
    Write-Silicon " Uptime: $($vm.Uptime)"
    Write-Silicon " Serial: $SerialPipe"
    Write-Silicon "═══════════════════════════════════════"
}

function Destroy-Silicon {
    Write-Silicon "DESTROYING VM '$VMName'..."
    Stop-Silicon
    Remove-VM -Name $VMName -Force -ErrorAction SilentlyContinue
    # Don't delete VHDX/kernel files
    Write-OK "VM removed. Files preserved in $SiliconDir"
}

# ============================================================================
# DISPATCH
# ============================================================================
switch ($Action.ToLower()) {
    "setup" {
        Ensure-HyperV
        Setup-Infrastructure
        Create-SiliconVM
        if ($KernelBin) { Install-Kernel -BinPath $KernelBin }
        Write-Silicon ""
        Write-Silicon "NEXT: Build kernel → .\hyperv_setup.ps1 -Action setup -KernelBin build\well_silicon.bin"
        Write-Silicon "THEN: .\hyperv_setup.ps1 -Action start"
    }
    "start"   { Start-Silicon }
    "stop"    { Stop-Silicon }
    "restart" { Restart-Silicon }
    "status"  { Get-SiliconStatus }
    "destroy" { Destroy-Silicon }
    "install" { Install-Kernel -BinPath $KernelBin }
    default   { Write-ERR "Unknown action: $Action. Use: setup|start|stop|restart|status|destroy|install" }
}
