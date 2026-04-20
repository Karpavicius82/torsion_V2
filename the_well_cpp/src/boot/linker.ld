/* ============================================================================
 * LINKER.LD — Living Silicon Bare-Metal Linker Script
 *
 * Memory layout:
 *   0x00100000 (1MB)  - kernel .text start
 *   .rodata, .data    - kernel read-only and initialized data
 *   .bss              - kernel zero-initialized data
 *   __heap_start      - bump allocator arena (256MB)
 * ============================================================================ */

ENTRY(_start)

SECTIONS {
    . = 1M;

    .boot : {
        KEEP(*(.multiboot))
    }

    .text ALIGN(4K) : {
        *(.text)
        *(.text.*)
    }

    .rodata ALIGN(4K) : {
        *(.rodata)
        *(.rodata.*)
    }

    .data ALIGN(4K) : {
        *(.data)
        *(.data.*)
    }

    .bss ALIGN(4K) : {
        __bss_start = .;
        *(.bss)
        *(COMMON)
        __bss_end = .;
    }

    . = ALIGN(4K);
    __heap_start = .;
    . = . + 256M;
    __heap_end = .;

    __kernel_end = .;

    /DISCARD/ : {
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }
}
