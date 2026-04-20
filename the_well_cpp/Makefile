# ============================================================================
# Makefile — Living Silicon Complete Build System
#
# Targets:
#   make kernel     — Build bare-metal kernel (well_silicon.bin)
#   make controller — Build Windows host controller (silicon_controller.exe)
#   make image      — Create bootable VHDX for Hyper-V
#   make deploy     — Full pipeline: kernel + image + VM setup
#   make boot       — Start Hyper-V VM
#   make stop       — Stop Hyper-V VM
#   make test       — Build + boot + run stress test
#   make hosted     — Build hosted-mode .exe (for development)
#   make clean      — Remove build artifacts
#
# Requirements:
#   Bare-metal: WSL2 with g++ and nasm (or x86_64-elf-gcc)
#   Controller: MSVC (cl.exe) or g++ on Windows
#   Deploy:     Hyper-V enabled, admin rights
# ============================================================================

# ── Paths ──
SRC       := src
BUILD     := build
DEPLOY    := deploy
SILICON   := C:/Silicon
VHDX      := $(SILICON)/well_silicon.vhdx
BIN       := $(BUILD)/well_silicon.bin

# ── Tools (WSL2 default) ──
WSL_DISTRO := Ubuntu-24.04
CXX_BARE   := wsl -d $(WSL_DISTRO) -- g++
AS_BARE     := wsl -d $(WSL_DISTRO) -- nasm
LD_BARE     := wsl -d $(WSL_DISTRO) -- ld

# ── Windows tools ──
CXX_WIN     := C:/msys64/ucrt64/bin/g++.exe
PS          := powershell.exe -ExecutionPolicy Bypass -File

# ── Flags ──
CXXFLAGS_BARE := -ffreestanding -fno-exceptions -fno-rtti -nostdlib \
                 -mno-red-zone -mavx2 -mfma -O3 -std=c++20 -DBARE_METAL -I$(SRC)
CXXFLAGS_HOST := -O3 -mavx2 -mfma -std=c++20 -I$(SRC)
LDFLAGS_BARE  := -n -T $(SRC)/boot/linker.ld

# ── Source lists ──
KERNEL_SRC  := $(SRC)/kernel.cpp
BOOT_SRC    := $(SRC)/boot/boot.asm
CTRL_SRC    := $(DEPLOY)/silicon_controller.cpp
MAIN_SRC    := $(SRC)/main.cpp

# ── Objects ──
KERNEL_O    := $(BUILD)/kernel.o
BOOT_O      := $(BUILD)/boot.o

# ============================================================================
# TARGETS
# ============================================================================

.PHONY: all kernel controller image deploy boot stop status test hosted clean help

all: kernel controller
	@echo "[SILICON] Build complete."

help:
	@echo ""
	@echo "  Living Silicon — Build System"
	@echo "  ═══════════════════════════════"
	@echo ""
	@echo "  make kernel      Build bare-metal kernel (.bin)"
	@echo "  make controller  Build Windows host controller (.exe)"
	@echo "  make image       Create bootable VHDX"
	@echo "  make deploy      Full pipeline (kernel + image + VM)"
	@echo "  make boot        Start Hyper-V VM"
	@echo "  make stop        Stop Hyper-V VM"
	@echo "  make status      Show VM status"
	@echo "  make test        Build + boot + stress test"
	@echo "  make hosted      Build hosted-mode dev binary"
	@echo "  make clean       Remove build artifacts"
	@echo ""

# ── Bare-metal kernel ──
kernel: $(BIN)
	@echo "[SILICON] Kernel: $(BIN) ($$(wc -c < $(BIN) 2>/dev/null || echo '???') bytes)"

$(BUILD):
	@mkdir -p $(BUILD) 2>/dev/null || mkdir $(BUILD)

$(KERNEL_O): $(KERNEL_SRC) | $(BUILD)
	@echo "[CXX] kernel.cpp → kernel.o (bare-metal)"
	$(CXX_BARE) $(CXXFLAGS_BARE) -c $(KERNEL_SRC) -o $(KERNEL_O)

$(BOOT_O): $(BOOT_SRC) | $(BUILD)
	@echo "[ASM] boot.asm → boot.o"
	$(AS_BARE) -f elf64 $(BOOT_SRC) -o $(BOOT_O)

$(BIN): $(BOOT_O) $(KERNEL_O)
	@echo "[LD]  → well_silicon.bin"
	$(LD_BARE) $(LDFLAGS_BARE) -o $(BIN) $(BOOT_O) $(KERNEL_O)

# ── Windows controller ──
controller: $(BUILD)/silicon_controller.exe
	@echo "[SILICON] Controller: $(BUILD)/silicon_controller.exe"

$(BUILD)/silicon_controller.exe: $(CTRL_SRC) | $(BUILD)
	@echo "[CXX] silicon_controller.cpp → silicon_controller.exe"
	$(CXX_WIN) -O2 -std=c++20 -o $@ $< -lkernel32

# ── Hosted mode (development) ──
hosted: $(BUILD)/the_well_cpp.exe
	@echo "[SILICON] Hosted: $(BUILD)/the_well_cpp.exe"

$(BUILD)/the_well_cpp.exe: $(MAIN_SRC) | $(BUILD)
	@echo "[CXX] main.cpp → the_well_cpp.exe (hosted)"
	$(CXX_WIN) $(CXXFLAGS_HOST) -o $@ $<

# ── VHDX image ──
image: kernel
	@echo "[IMAGE] Creating bootable VHDX..."
	$(PS) $(DEPLOY)/build_image.ps1 -SkipVHDX:$$false

# ── Deploy (full pipeline) ──
deploy: kernel controller image
	@echo "[DEPLOY] Setting up Hyper-V VM..."
	$(PS) $(DEPLOY)/hyperv_setup.ps1 -Action setup -KernelBin $(BIN)

# ── VM control ──
boot:
	$(PS) $(DEPLOY)/hyperv_setup.ps1 -Action start

stop:
	$(PS) $(DEPLOY)/hyperv_setup.ps1 -Action stop

status:
	$(PS) $(DEPLOY)/hyperv_setup.ps1 -Action status

# ── Full test pipeline ──
test: deploy boot
	@echo "[TEST] Waiting for kernel boot (5s)..."
	@sleep 5 2>/dev/null || timeout /t 5 /nobreak >nul
	@echo "[TEST] Launching controller..."
	$(BUILD)/silicon_controller.exe --csv $(SILICON)/results.csv
	@echo "[TEST] Results: $(SILICON)/results.csv"

# ── Clean ──
clean:
	@echo "[CLEAN] Removing build artifacts..."
	rm -rf $(BUILD)/*.o $(BUILD)/*.bin $(BUILD)/*.exe 2>/dev/null || del /q $(BUILD)\*.o $(BUILD)\*.bin $(BUILD)\*.exe 2>nul
	@echo "[CLEAN] Done."
