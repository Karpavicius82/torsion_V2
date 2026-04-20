; ============================================================================
; BOOT.ASM — Multiboot2 Bootloader Entry Point for Living Silicon
;
; Loaded by GRUB/Multiboot → sets up 64-bit long mode → jumps to C++ kernel
;
; Memory map:
;   0x00100000 - kernel start (1MB)
;   0x00200000 - heap start (2MB)
;   0x10200000 - heap end   (258MB)
;   0xB8000    - VGA text buffer
; ============================================================================

section .multiboot
align 8

; Multiboot2 header
MB2_MAGIC  equ 0xE85250D6
MB2_ARCH   equ 0           ; i386
MB2_LENGTH equ (mb2_end - mb2_start)
MB2_CHECKSUM equ -(MB2_MAGIC + MB2_ARCH + MB2_LENGTH)

mb2_start:
    dd MB2_MAGIC
    dd MB2_ARCH
    dd MB2_LENGTH
    dd MB2_CHECKSUM
    ; End tag
    dw 0    ; type
    dw 0    ; flags
    dd 8    ; size
mb2_end:

section .bss
align 4096

; Page tables for identity mapping
p4_table: resb 4096
p3_table: resb 4096
p2_table: resb 4096

; Stack
stack_bottom: resb 65536  ; 64KB kernel stack
stack_top:

section .text
global _start
extern kernel_main

bits 32

_start:
    ; Set up stack
    mov esp, stack_top

    ; Save multiboot info pointer
    push ebx
    push eax

    ; Check CPUID support
    pushfd
    pop eax
    mov ecx, eax
    xor eax, 1 << 21
    push eax
    popfd
    pushfd
    pop eax
    push ecx
    popfd
    cmp eax, ecx
    je .no_cpuid

    ; Check long mode support
    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jz .no_long_mode

    ; Set up paging for long mode
    call setup_page_tables
    call enable_paging

    ; Load 64-bit GDT
    lgdt [gdt64.pointer]

    ; Far jump to 64-bit code
    jmp gdt64.code:long_mode_start

.no_cpuid:
.no_long_mode:
    ; Fallback: print error to VGA
    mov dword [0xB8000], 0x4F524F45  ; "ER"
    mov dword [0xB8004], 0x4F3A4F52  ; "R:"
    mov dword [0xB8008], 0x4F364F36  ; "64"
    hlt

setup_page_tables:
    ; Map P4[0] -> P3
    mov eax, p3_table
    or eax, 0b11    ; present + writable
    mov [p4_table], eax

    ; Map P3[0] -> P2
    mov eax, p2_table
    or eax, 0b11
    mov [p3_table], eax

    ; Map P2[0..511] -> 0..1GB (2MB pages)
    mov ecx, 0
.map_p2:
    mov eax, 0x200000   ; 2MB
    mul ecx
    or eax, 0b10000011  ; present + writable + huge
    mov [p2_table + ecx * 8], eax
    inc ecx
    cmp ecx, 512
    jne .map_p2
    ret

enable_paging:
    ; Load P4 to CR3
    mov eax, p4_table
    mov cr3, eax

    ; Enable PAE
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    ; Set long mode bit in EFER MSR
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr

    ; Enable paging
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax
    ret

; 64-bit GDT
section .rodata
align 8
gdt64:
    dq 0                         ; null entry
.code: equ $ - gdt64
    dq (1<<43)|(1<<44)|(1<<47)|(1<<53) ; code segment
.data: equ $ - gdt64
    dq (1<<44)|(1<<47)|(1<<41)        ; data segment
.pointer:
    dw $ - gdt64 - 1    ; limit
    dq gdt64             ; base

section .text
bits 64

long_mode_start:
    ; Set up data segments
    mov ax, gdt64.data
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov fs, ax
    mov gs, ax

    ; Re-set stack in 64-bit mode
    mov rsp, stack_top

    ; Enable SSE/AVX
    ; CR0: clear EM, set MP
    mov rax, cr0
    and ax, 0xFFFB
    or ax, 0x2
    mov cr0, rax

    ; CR4: set OSFXSR, OSXMMEXCPT, OSXSAVE
    mov rax, cr4
    or eax, (1 << 9) | (1 << 10) | (1 << 18)
    mov cr4, rax

    ; XCR0: enable SSE + AVX + AVX-512
    xor ecx, ecx
    xgetbv
    or eax, 0b111  ; SSE + AVX state
    xsetbv

    ; Jump to C++ kernel
    call kernel_main

    ; Halt if kernel returns
.halt:
    cli
    hlt
    jmp .halt
