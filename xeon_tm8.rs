rust
#![feature(stdsimd, asm_const, naked_functions)]
#![allow(unused, internal_features)]

use std::arch::x86_64::*;
use std::arch::asm;

// ============================================================================
// STRUCTS - Cache-line aligned, padded for prefetch friendliness
// ============================================================================

/// int2048 as 8(256) limbs 4(u64), stored little-endian
/// 
/// Memory span spans precisely 4 cache lines
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct Int2048 {
    pub limbs: [[u64; 4]; 8],  // 256 bytes
}

/// 4096-bit result - 8 cache lines
#[repr(C, align(64))]
#[derive(Clone, Copy)]  
pub struct Int4096 {
    pub limbs: [[u64; 4]; 16], // 512 bytes
}

/// Workspace for intermediates during eval w overflow
#[repr(C, align(64))]
pub struct Toom8Workspace {
    // may overflow to ~320 bits due to coefficient scalarity
    // Store as 6 × u64 (384 bits) to be safe, padded to 64 bytes for alignment
    pub eval_a: [[u64; 8]; 15],     // 15 × 64 = 960 bytes
    pub eval_b: [[u64; 8]; 15],     // 960 bytes
    pub products: [[u64; 16]; 15],  // 15 × 128 = 1920 bytes
    // Interpolation scratch
    pub interp: [[u64; 16]; 15],    // 1920 bytes
    // Carry propagation
    pub carries: [[u64; 4]; 32],    // 1024 bytes
}

impl Toom8Workspace {
    pub fn new() -> Box<Self> {
        // heap allocation with alignment
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc_zeroed(layout) as *mut Self;
            Box::from_raw(ptr)
        }
    }
}

// ============================================================================
// TEXTBOOKISH TM8 EVALUATION POINTS
// ============================================================================

// Alternative cf Bodrato accept larger coefficients to avoids the scaling complexity.

/// Evaluation coefficients for point i, coefficient j
/// eval_point[i][j] = point_i^j
/// 
/// Points: 0, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, inf
/// (15 points for degree-14 result polynomial)
pub const EVAL_COEFFS: [[i64; 8]; 15] = [
    // Point 0: [1, 0, 0, 0, 0, 0, 0, 0]
    [1, 0, 0, 0, 0, 0, 0, 0],
    // Point 1: powers of 1
    [1, 1, 1, 1, 1, 1, 1, 1],
    // Point -1: powers of -1
    [1, -1, 1, -1, 1, -1, 1, -1],
    // Point 2: powers of 2
    [1, 2, 4, 8, 16, 32, 64, 128],
    // Point -2
    [1, -2, 4, -8, 16, -32, 64, -128],
    // Point 4
    [1, 4, 16, 64, 256, 1024, 4096, 16384],
    // Point -4
    [1, -4, 16, -64, 256, -1024, 4096, -16384],
    // Point 8
    [1, 8, 64, 512, 4096, 32768, 262144, 2097152],
    // Point -8
    [1, -8, 64, -512, 4096, -32768, 262144, -2097152],
    // Point 16
    [1, 16, 256, 4096, 65536, 1048576, 16777216, 268435456],
    // Point -16
    [1, -16, 256, -4096, 65536, -1048576, 16777216, -268435456],
    // Point 32
    [1, 32, 1024, 32768, 1048576, 33554432, 1073741824, 34359738368],
    // Point -32
    [1, -32, 1024, -32768, 1048576, -33554432, 1073741824, -34359738368],
    // Point 64
    [1, 64, 4096, 262144, 16777216, 1073741824, 68719476736, 4398046511104],
    // Point inf: [0, 0, 0, 0, 0, 0, 0, 1] (leading coefficient only)
    [0, 0, 0, 0, 0, 0, 0, 1],
];

// ============================================================================
// MICRO-OPTIMIZATION ON XEON
// ============================================================================
//
// Pre Xeon 6 considerations:
// 1. 512-bit operations can cause frequency throttling (varies by SKU/config)
// 2. Some configs execute zmm as 2×ymm on ports 0+1 or 0+5
// Granite Rapids (Xeon 6):
// 1. True 512-bit,no throttling
// 2. benefits from the half-independence for superscalar dispatch
//
// Strategy: Treat zmm as "2 ymm stapled together" in the dependency graph.
// Operations on lanes [0:3] should not depend on lanes [4:7] within same zmm.
//
// Register strategy:
// - zmm0-7:   Operand A perduring eval
// - zmm8-15:  Operand B perduring eval 
// - zmm16-23: Accumulators
// - zmm24-31: Sidecar
//
// The sidecar shifts between accumulation, intermediate products, and matrix interpolation 
// ============================================================================

// ============================================================================
// PHASE 1: LOAD WITH PREFETCH CHOREOGRAPHY
// ============================================================================

/// Load operands w soft prefetch
#[inline(always)]
pub unsafe fn load_operands_with_prefetch(
    a: *const Int2048,
    b: *const Int2048,
    ws: *mut Toom8Workspace,
) {
    asm!(
        // Prefetch workspace cache lines
        // T0 = prefetch to all cache levels (L1/L2/L3)
        "prefetcht0 [{ws}]",
        "prefetcht0 [{ws} + 64]",
        "prefetcht0 [{ws} + 128]",
        "prefetcht0 [{ws} + 192]",
        
        // Load A's 8 limbs into zmm0-7
        // Using ymm loads (256-bit) which zero-extend to zmm
        // This is fine on both Ice Lake and Granite Rapids
        "vmovdqu64 ymm0, [{a}]",
        "vmovdqu64 ymm1, [{a} + 32]",
        
        // Interleave more prefetches during load stalls
        "prefetcht0 [{ws} + 256]",
        "prefetcht0 [{ws} + 320]",
        
        "vmovdqu64 ymm2, [{a} + 64]",
        "vmovdqu64 ymm3, [{a} + 96]",
        
        "prefetcht0 [{ws} + 384]",
        "prefetcht0 [{ws} + 448]",
        
        "vmovdqu64 ymm4, [{a} + 128]",
        "vmovdqu64 ymm5, [{a} + 160]",
        "vmovdqu64 ymm6, [{a} + 192]",
        "vmovdqu64 ymm7, [{a} + 224]",
        
        // Load B's 8 limbs into zmm8-15
        "vmovdqu64 ymm8, [{b}]",
        "vmovdqu64 ymm9, [{b} + 32]",
        "vmovdqu64 ymm10, [{b} + 64]",
        "vmovdqu64 ymm11, [{b} + 96]",
        "vmovdqu64 ymm12, [{b} + 128]",
        "vmovdqu64 ymm13, [{b} + 160]",
        "vmovdqu64 ymm14, [{b} + 192]",
        "vmovdqu64 ymm15, [{b} + 224]",
        
        a = in(reg) a,
        b = in(reg) b,
        ws = in(reg) ws,
        // Don't clobber the zmm regs we just loaded
        options(nostack, preserves_flags),
    );
}

// ============================================================================
// PHASE 2: EVALUATION KERNEL
// ============================================================================
// 
// Most entries are powers of 2, so we can use shifts instead of multiplies.
//
// For each evaluation point p, compute:
//   eval_a[p] = a0 + a1*p + a2*p² + ... + a7*p⁷
//
// Horner's form:
//   eval_a[p] = a0 + p*(a1 + p*(a2 + p*(a3 + p*(a4 + p*(a5 + p*(a6 + p*a7))))))
//
// HOWEVER: Horner has a serial dependency chain of depth 7.
// Alternative direct summation tree reduction.
// 
// Direct summation for point 2:
//   a0 + (a1<<1) + (a2<<2) + (a3<<3) + (a4<<4) + (a5<<5) + (a6<<6) + (a7<<7)
//
/// 
/// if negate_odd = T negate odd-indexed terms for negative points
#[inline(always)]
pub unsafe fn eval_power_of_2_point(
    point_log2: u8,
    negate_odd: bool,
    out_a: *mut [u64; 8],
    out_b: *mut [u64; 8],
) {
    // obvs incomplete 
    if negate_odd {
        asm!(
            
            // immediate if unrolling per-point
            
            // --- Operand A evaluation ---
            // zmm0-7 contain a0..a7
            // Need: a0 - (a1<<k) + (a2<<2k) - (a3<<3k) + ...
            
            // Compute all shifts in parallel 
            "vpsllq zmm16, zmm1, {shift}",      // a1 << k
            "vpsllq zmm17, zmm2, {shift2}",     // a2 << 2k  
            "vpsllq zmm18, zmm3, {shift3}",     // a3 << 3k
            "vpsllq zmm19, zmm4, {shift4}",     // a4 << 4k
            "vpsllq zmm20, zmm5, {shift5}",     // a5 << 5k
            "vpsllq zmm21, zmm6, {shift6}",     // a6 << 6k
            "vpsllq zmm22, zmm7, {shift7}",     // a7 << 7k
            
            // Tree reduction alternating signature
            // Level 1: 4 parallel operations
            "vpsubq zmm16, zmm0, zmm16",        // a0 - a1<<k
            "vpsubq zmm17, zmm17, zmm18",       // a2<<2k - a3<<3k
            "vpsubq zmm18, zmm19, zmm20",       // a4<<4k - a5<<5k
            "vpsubq zmm19, zmm21, zmm22",       // a6<<6k - a7<<7k
            
            // Level 2: 2 parallel operations  
            "vpaddq zmm16, zmm16, zmm17",       // (a0 - a1<<k) + (a2<<2k - a3<<3k)
            "vpaddq zmm17, zmm18, zmm19",       // (a4<<4k - a5<<5k) + (a6<<6k - a7<<7k)
            
            // Level 3: final sum
            "vpaddq zmm16, zmm16, zmm17",       // complete evaluation for A
            
            // Store result (with overflow space)
            "vmovdqu64 [{out_a}], zmm16",
            
            // --- Operand B evaluation (parallel with A on second execution unit) ---
            // zmm8-15 contain b0..b7
            // Use zmm24-31 for B's computation (sidecar bank)
            
            "vpsllq zmm24, zmm9, {shift}",
            "vpsllq zmm25, zmm10, {shift2}",
            "vpsllq zmm26, zmm11, {shift3}",
            "vpsllq zmm27, zmm12, {shift4}",
            "vpsllq zmm28, zmm13, {shift5}",
            "vpsllq zmm29, zmm14, {shift6}",
            "vpsllq zmm30, zmm15, {shift7}",
            
            "vpsubq zmm24, zmm8, zmm24",
            "vpsubq zmm25, zmm25, zmm26",
            "vpsubq zmm26, zmm27, zmm28",
            "vpsubq zmm27, zmm29, zmm30",
            
            "vpaddq zmm24, zmm24, zmm25",
            "vpaddq zmm25, zmm26, zmm27",
            
            "vpaddq zmm24, zmm24, zmm25",
            
            "vmovdqu64 [{out_b}], zmm24",
            
            shift = const point_log2 as i32,
            shift2 = const (point_log2 * 2) as i32,
            shift3 = const (point_log2 * 3) as i32,
            shift4 = const (point_log2 * 4) as i32,
            shift5 = const (point_log2 * 5) as i32,
            shift6 = const (point_log2 * 6) as i32,
            shift7 = const (point_log2 * 7) as i32,
            out_a = in(reg) out_a,
            out_b = in(reg) out_b,
            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _,
            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
            out("zmm28") _, out("zmm29") _, out("zmm30") _,
            options(nostack, preserves_flags),
        );
    } else {
        asm!(
            "vpsllq zmm16, zmm1, {shift}",
            "vpsllq zmm17, zmm2, {shift2}",
            "vpsllq zmm18, zmm3, {shift3}",
            "vpsllq zmm19, zmm4, {shift4}",
            "vpsllq zmm20, zmm5, {shift5}",
            "vpsllq zmm21, zmm6, {shift6}",
            "vpsllq zmm22, zmm7, {shift7}",
            
            // All additions for positive point
            "vpaddq zmm16, zmm0, zmm16",
            "vpaddq zmm17, zmm17, zmm18",
            "vpaddq zmm18, zmm19, zmm20",
            "vpaddq zmm19, zmm21, zmm22",
            
            "vpaddq zmm16, zmm16, zmm17",
            "vpaddq zmm17, zmm18, zmm19",
            
            "vpaddq zmm16, zmm16, zmm17",
            
            "vmovdqu64 [{out_a}], zmm16",
            
            // Operand B
            "vpsllq zmm24, zmm9, {shift}",
            "vpsllq zmm25, zmm10, {shift2}",
            "vpsllq zmm26, zmm11, {shift3}",
            "vpsllq zmm27, zmm12, {shift4}",
            "vpsllq zmm28, zmm13, {shift5}",
            "vpsllq zmm29, zmm14, {shift6}",
            "vpsllq zmm30, zmm15, {shift7}",
            
            "vpaddq zmm24, zmm8, zmm24",
            "vpaddq zmm25, zmm25, zmm26",
            "vpaddq zmm26, zmm27, zmm28",
            "vpaddq zmm27, zmm29, zmm30",
            
            "vpaddq zmm24, zmm24, zmm25",
            "vpaddq zmm25, zmm26, zmm27",
            
            "vpaddq zmm24, zmm24, zmm25",
            
            "vmovdqu64 [{out_b}], zmm24",
            
            shift = const point_log2 as i32,
            shift2 = const (point_log2 * 2) as i32,
            shift3 = const (point_log2 * 3) as i32,
            shift4 = const (point_log2 * 4) as i32,
            shift5 = const (point_log2 * 5) as i32,
            shift6 = const (point_log2 * 6) as i32,
            shift7 = const (point_log2 * 7) as i32,
            out_a = in(reg) out_a,
            out_b = in(reg) out_b,
            out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
            out("zmm20") _, out("zmm21") _, out("zmm22") _,
            out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
            out("zmm28") _, out("zmm29") _, out("zmm30") _,
            options(nostack, preserves_flags),
        );
    }
}

/// Special case: evaluate pt 0 (ret limb[0])
#[inline(always)]
pub unsafe fn eval_point_0(out_a: *mut [u64; 8], out_b: *mut [u64; 8]) {
    asm!(
        "vmovdqu64 [{out_a}], zmm0",
        "vmovdqu64 [{out_b}], zmm8",
        out_a = in(reg) out_a,
        out_b = in(reg) out_b,
        options(nostack, preserves_flags),
    );
}

/// Special case: evaluate at point (inf) (ret limb[7])
#[inline(always)]
pub unsafe fn eval_point_inf(out_a: *mut [u64; 8], out_b: *mut [u64; 8]) {
    asm!(
        "vmovdqu64 [{out_a}], zmm7",
        "vmovdqu64 [{out_b}], zmm15",
        out_a = in(reg) out_a,
        out_b = in(reg) out_b,
        options(nostack, preserves_flags),
    );
}

/// Special case: eval at pt 1 (sum limbs)
#[inline(always)]
pub unsafe fn eval_point_1(out_a: *mut [u64; 8], out_b: *mut [u64; 8]) {
    asm!(
        "vpaddq zmm16, zmm0, zmm1",
        "vpaddq zmm17, zmm2, zmm3",
        "vpaddq zmm18, zmm4, zmm5",
        "vpaddq zmm19, zmm6, zmm7",
        "vpaddq zmm24, zmm8, zmm9",
        "vpaddq zmm25, zmm10, zmm11",
        "vpaddq zmm26, zmm12, zmm13",
        "vpaddq zmm27, zmm14, zmm15",
        
        "vpaddq zmm16, zmm16, zmm17",
        "vpaddq zmm17, zmm18, zmm19",
        "vpaddq zmm24, zmm24, zmm25",
        "vpaddq zmm25, zmm26, zmm27",
        
        "vpaddq zmm16, zmm16, zmm17",
        "vpaddq zmm24, zmm24, zmm25",
        
        "vmovdqu64 [{out_a}], zmm16",
        "vmovdqu64 [{out_b}], zmm24",
        
        out_a = in(reg) out_a,
        out_b = in(reg) out_b,
        out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
        out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
        options(nostack, preserves_flags),
    );
}

/// Special case: evaluate at point -1 (alternating summation)
#[inline(always)]
pub unsafe fn eval_point_neg1(out_a: *mut [u64; 8], out_b: *mut [u64; 8]) {
    asm!(
        // a0 - a1 + a2 - a3 + a4 - a5 + a6 - a7
        // Regroup: (a0 - a1) + (a2 - a3) + (a4 - a5) + (a6 - a7)
        
        "vpsubq zmm16, zmm0, zmm1",
        "vpsubq zmm17, zmm2, zmm3",
        "vpsubq zmm18, zmm4, zmm5",
        "vpsubq zmm19, zmm6, zmm7",
        
        "vpsubq zmm24, zmm8, zmm9",
        "vpsubq zmm25, zmm10, zmm11",
        "vpsubq zmm26, zmm12, zmm13",
        "vpsubq zmm27, zmm14, zmm15",
        
        "vpaddq zmm16, zmm16, zmm17",
        "vpaddq zmm17, zmm18, zmm19",
        "vpaddq zmm24, zmm24, zmm25",
        "vpaddq zmm25, zmm26, zmm27",
        
        "vpaddq zmm16, zmm16, zmm17",
        "vpaddq zmm24, zmm24, zmm25",
        
        "vmovdqu64 [{out_a}], zmm16",
        "vmovdqu64 [{out_b}], zmm24",
        
        out_a = in(reg) out_a,
        out_b = in(reg) out_b,
        out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
        out("zmm24") _, out("zmm25") _, out("zmm26") _, out("zmm27") _,
        options(nostack, preserves_flags),
    );
}

// ============================================================================
// PHASE 3: POINTWISE MULTIPLICATION
// ============================================================================
// Each mult ~640b
// - With AVX512-IFMA: 52-bit limbs, use VPMADD52LUQ/VPMADD52HUQ
// - Without IFMA: decompose 64×64 into 32×32 using VPMULUDQ
//

/// Check for IFMA support at runtime
#[inline(always)]
pub fn has_ifma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::__cpuid_count;
        unsafe {
            let result = __cpuid_count(7, 0);
            (result.ebx & (1 << 21)) != 0  // AVX512-IFMA bit
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    { false }
}

/// Each 64-bit limb is treated as 2×32-bit for multiplication,
/// then we reconstruct the full 128-bit partial products.
///
/// Input:  a[0..4], b[0..4] as u64 arrays
/// Output: r[0..8] as u64 array
///
#[inline(always)]
pub unsafe fn mul_256x256_muludq(
    a_ptr: *const [u64; 8],  // only first 4 used, rest is padding
    b_ptr: *const [u64; 8],
    r_ptr: *mut [u64; 16],   // only first 8 used
) {
    // The full schoolbook is long
    asm!(
        // Load a and b with broadcast preparation
        "vmovdqu64 ymm0, [{a}]",        // a[0..3]
        "vmovdqu64 ymm8, [{b}]",        // b[0..3]
        
      
        // 1. Create low version: mask out high 32b/qword
        // 2. Create high version: shift right by 32
        // 3. Do 4 VPMULUDQ per pair of limbs, accumulate with proper shifts
        
        // Mask for low 32 bits of each qword
        "mov rax, 0x00000000FFFFFFFF",
        "vpbroadcastq zmm31, rax",      // zmm31 = [mask, mask, mask, ...]
        
        // Extract low halves
        "vpandq zmm1, zmm0, zmm31",     // a_lo
        "vpandq zmm9, zmm8, zmm31",     // b_lo
        
        // Extract high halves  
        "vpsrlq zmm2, zmm0, 32",        // a_hi
        "vpsrlq zmm10, zmm8, 32",       // b_hi
        
        // === Compute partial products ===
        // For each (a_i, b_j) pair, we get 4 products:
        //   a_i_lo * b_j_lo  -> contributes at position 2i+2j
        //   a_i_lo * b_j_hi  -> contributes at position 2i+2j + 32 bits
        //   a_i_hi * b_j_lo  -> contributes at position 2i+2j + 32 bits
        //   a_i_hi * b_j_hi  -> contributes at position 2i+2j + 64 bits
        
        // For position 0: only a0*b0 contributes
        // a0_lo * b0_lo -> pos 0
        // (a0_lo * b0_hi + a0_hi * b0_lo) << 32 -> overlaps pos 0 and 1
        // a0_hi * b0_hi -> pos 1
        
        // Broadcast a0 components for multiplication with all b components
        "vpbroadcastq zmm3, xmm1",       // a0_lo broadcast
        "vpbroadcastq zmm4, xmm2",       // a0_hi broadcast
        
        // a0_lo * b_lo
        "vpmuludq zmm16, zmm3, zmm9",    // [a0_lo*b0_lo, a0_lo*b1_lo, a0_lo*b2_lo, a0_lo*b3_lo]
        
        // a0_lo * b_hi
        "vpmuludq zmm17, zmm3, zmm10",   // [a0_lo*b0_hi, ...]
        
        // a0_hi * b_lo
        "vpmuludq zmm18, zmm4, zmm9",    // [a0_hi*b0_lo, ...]
        
        // a0_hi * b_hi
        "vpmuludq zmm19, zmm4, zmm10",   // [a0_hi*b0_hi, ...]
        
        // Middle products (zmm17, zmm18) contribute to position + 32 bits
        // Add them together
        "vpaddq zmm17, zmm17, zmm18",
        
        // Split middle sum into low and high 32-bit parts for carry
        "vpandq zmm18, zmm17, zmm31",    // middle_lo
        "vpsrlq zmm17, zmm17, 32",       // middle_hi
        
        // Shift middle_lo left by 32 and add to low product
        "vpsllq zmm18, zmm18, 32",
        "vpaddq zmm16, zmm16, zmm18",
        
        // Add middle_hi to high product
        "vpaddq zmm19, zmm19, zmm17",
        
        // Now zmm16 has low 64 bits of each a0*b_j product
        // zmm19 has high 64 bits of each a0*b_j product
        
        // These need to be accumulated at the right positions:
        // zmm16[0] (a0*b0 low)  -> r[0]
        // zmm16[1] (a0*b1 low)  -> r[1]
        // zmm16[2] (a0*b2 low)  -> r[2]
        // zmm16[3] (a0*b3 low)  -> r[3]
        // zmm19[0] (a0*b0 high) -> r[1]
        // zmm19[1] (a0*b1 high) -> r[2]
        // etc.
        
        // ... Continue for a1, a2, a3 ...
        
        // Store result
        "vmovdqu64 [{r}], zmm16",        // 
        // Placeholder
        
        a = in(reg) a_ptr,
        b = in(reg) b_ptr,
        r = in(reg) r_ptr,
        out("rax") _,
        out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _, out("zmm4") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _,
        out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
        out("zmm31") _,
        options(nostack, preserves_flags),
    );
}

/// Product = 10 limbs before normalization
#[target_feature(enable = "avx512ifma")]
#[inline(always)]
pub unsafe fn mul_256x256_ifma(
    a_ptr: *const [u64; 8],  // packed as 52-bit limbs
    b_ptr: *const [u64; 8],
    r_ptr: *mut [u64; 16],
) {
    // VPMADD52LUQ: dst += (a * b)[51:0] lo
    // VPMADD52HUQ: dst += (a * b)[103:52] hi
    
    asm!(
        // Load 5a 5b
        "vmovdqu64 zmm0, [{a}]",
        "vmovdqu64 zmm8, [{b}]",
        
        // Initialize 10 accumulators
        "vpxorq zmm16, zmm16, zmm16",    // r0
        "vpxorq zmm17, zmm17, zmm17",    // r1
        "vpxorq zmm18, zmm18, zmm18",    // r2
        "vpxorq zmm19, zmm19, zmm19",    // r3
        "vpxorq zmm20, zmm20, zmm20",    // r4
        "vpxorq zmm21, zmm21, zmm21",    // r5
        "vpxorq zmm22, zmm22, zmm22",    // r6
        "vpxorq zmm23, zmm23, zmm23",    // r7
        "vpxorq zmm24, zmm24, zmm24",    // r8
        "vpxorq zmm25, zmm25, zmm25",    // r9
        
        // Broadcast a[0] and multiply by all b[j]
        "vpbroadcastq zmm1, xmm0",
        "vpmadd52luq zmm16, zmm1, zmm8", // r[0..4] += low(a0 * b[0..4])
        "vpmadd52huq zmm17, zmm1, zmm8", // r[1..5] += high(a0 * b[0..4])
        
        // Extract and broadcast a[1]
        "valignq zmm2, zmm0, zmm0, 1",   // rotate to get a[1] in position 0
        "vpbroadcastq zmm1, xmm2",
        "vpmadd52luq zmm17, zmm1, zmm8", // r[1..5] += low(a1 * b[0..4])
        "vpmadd52huq zmm18, zmm1, zmm8", // r[2..6] += high(a1 * b[0..4])
        
        // ... continue pattern for a[2], a[3], a[4] ...
        
        // Store results (still need carry normalization)
        "vmovdqu64 [{r}], zmm16",
        "vmovdqu64 [{r} + 64], zmm20",
        
        a = in(reg) a_ptr,
        b = in(reg) b_ptr,
        r = in(reg) r_ptr,
        out("zmm0") _, out("zmm1") _, out("zmm2") _,
        out("zmm8") _,
        out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
        out("zmm20") _, out("zmm21") _, out("zmm22") _, out("zmm23") _,
        out("zmm24") _, out("zmm25") _,
        options(nostack, preserves_flags),
    );
}

// ============================================================================
// PHASE 4: INTERPOLATION
// ============================================================================
//
// Given 15 pointwise products p[0..14], recover the 15 coefficients c[0..14]
// of the product polynomial.
//
// This is the inverse of the evaluation matrix.
// - Many entries are 0; ~0 entries are rational small Z
// - Factorable to simpler operations cf Bodrato)
//
// For Toom-8 with power-of-2 points, the interpolation involves:
// 1. Compute symmetric/antisymmetric sums: s_i = p[2i] + p[2i+1], d_i = p[2i] - p[2i+1]
// 2. Apply smaller interpolation matrices to s and d independently
// 3. Exact divisions by powers of 2 (just shifts)
// 4. Exact divisions by small odd numbers (multiply by modular inverse + shift)

/// Interpolation step 1: Compute symmetric and antisymmetric components
#[inline(always)]
pub unsafe fn interp_sym_antisym(ws: *mut Toom8Workspace) {
    // p[0] and p[14] (∞) don't have pairs
    // For points ±1, ±2, ±4, etc:
    //   s = p[+k] + p[-k]  (symmetric)
    //   d = p[+k] - p[-k]  (antisymmetric)
    
    asm!(
        // Load product pairs and compute sums/differences
        // Point ±1: products at indices 1 and 2
        "vmovdqu64 zmm0, [{ws} + {p1_off}]",
        "vmovdqu64 zmm1, [{ws} + {p2_off}]",
        "vpaddq zmm16, zmm0, zmm1",      // s1 = p[1] + p[2]
        "vpsubq zmm17, zmm0, zmm1",      // d1 = p[1] - p[2]
        
        // Point ±2: products at indices 3 and 4
        "vmovdqu64 zmm2, [{ws} + {p3_off}]",
        "vmovdqu64 zmm3, [{ws} + {p4_off}]",
        "vpaddq zmm18, zmm2, zmm3",      // s2
        "vpsubq zmm19, zmm2, zmm3",      // d2
        
        // ... continue for remaining pairs ...
        
        // Store symmetric components
        "vmovdqu64 [{ws} + {interp_off}], zmm16",
        "vmovdqu64 [{ws} + {interp_off} + 64], zmm18",
        
        ws = in(reg) ws,
        // Offsets into workspace (would be const in real code)
        p1_off = const 0usize,    // placeholder
        p2_off = const 128usize,
        p3_off = const 256usize,
        p4_off = const 384usize,
        interp_off = const 3840usize,  // offset to interp array
        out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
        out("zmm16") _, out("zmm17") _, out("zmm18") _, out("zmm19") _,
        options(nostack, preserves_flags),
    );
}

/// Exact division by small odd constant (for interpolation)
/// Uses the identity: a / k = a * k^(-1) mod 2^64, then shift if needed
/// 
/// For example, to divide by 3:
///   a / 3 = a * 0xAAAAAAAAAAAAAAAB (mod 2^64) works when a is divisible by 3
#[inline(always)]
pub unsafe fn exact_div_by_3_inplace(val: *mut [u64; 16]) {
    const INV3: u64 = 0xAAAA_AAAA_AAAA_AAAB;  // 3^(-1) mod 2^64
    
    asm!(
        "vpbroadcastq zmm31, [{inv3}]",
        "vmovdqu64 zmm0, [{val}]",
        "vmovdqu64 zmm1, [{val} + 64]",
        
        // Multiply by modular inverse
        // VPMULLQ does 64×64 → 64 (low 64 bits only) - exactly what we need
        "vpmullq zmm0, zmm0, zmm31",
        "vpmullq zmm1, zmm1, zmm31",
        
        "vmovdqu64 [{val}], zmm0",
        "vmovdqu64 [{val} + 64], zmm1",
        
        val = in(reg) val,
        inv3 = in(reg) &INV3,
        out("zmm0") _, out("zmm1") _, out("zmm31") _,
        options(nostack, preserves_flags),
    );
}

// ============================================================================
// PHASE 5: RECOMPOSITION
// ============================================================================
//
/// Propagate carries through the final result
#[inline(always)]
pub unsafe fn propagate_carries(ws: *mut Toom8Workspace, result: *mut Int4096) {
    // The 15 coefficients c[0..14] each occupy positions:
    
    asm!(
        // Start with c[0] in result[0..1]
        "vmovdqu64 zmm0, [{ws} + {c0_off}]",      // c[0] low 512 bits
        "vmovdqu64 [{result}], zmm0",       

        // Add c[1] to result[1..2], propagate carry
        "vmovdqu64 zmm1, [{ws} + {c1_off}]",
        "vmovdqu64 zmm0, [{result} + 64]",         // load result[1]
        "vpaddq zmm0, zmm0, zmm1",                 // result[1] += c[1]_low
        
        // Check for carry (overflow detection is cluster)
        // comp to one of the addenda
        // If result < addend, carry occurred
        "vpcmpuq k1, zmm0, zmm1, 1",               // k1 = (result < c[1]) ? 1 : 0
        
        // ... carry propagation continues ...
        
        ws = in(reg) ws,
        result = in(reg) result,
        c0_off = const 0usize,
        c1_off = const 128usize,
        out("zmm0") _, out("zmm1") _,
        out("k1") _,
        options(nostack, preserves_flags),
    );
}

// ============================================================================
// TOP-LEVEL ORCHESTRATION
// ============================================================================

/// Main multiplication entry point with runtime feature detection
pub fn toom8_mul_2048(a: &Int2048, b: &Int2048, result: &mut Int4096) {
    let mut ws = Toom8Workspace::new();
    
    unsafe {
        // Phase 1: Load operands
        load_operands_with_prefetch(a as *const _, b as *const _, ws.as_mut() as *mut _);
        
        // Phase 2: Evaluation
        eval_point_0(ws.eval_a[0].as_mut_ptr(), ws.eval_b[0].as_mut_ptr());
        eval_point_1(ws.eval_a[1].as_mut_ptr(), ws.eval_b[1].as_mut_ptr());
        eval_point_neg1(ws.eval_a[2].as_mut_ptr(), ws.eval_b[2].as_mut_ptr());
        eval_power_of_2_point(1, false, ws.eval_a[3].as_mut_ptr(), ws.eval_b[3].as_mut_ptr()); // point 2
        eval_power_of_2_point(1, true, ws.eval_a[4].as_mut_ptr(), ws.eval_b[4].as_mut_ptr());  // point -2
        eval_power_of_2_point(2, false, ws.eval_a[5].as_mut_ptr(), ws.eval_b[5].as_mut_ptr()); // point 4
        eval_power_of_2_point(2, true, ws.eval_a[6].as_mut_ptr(), ws.eval_b[6].as_mut_ptr());  // point -4
        // ... remaining points ...
        eval_point_inf(ws.eval_a[14].as_mut_ptr(), ws.eval_b[14].as_mut_ptr());
        
        // Phase 3: Pointwise multiplication
        // Runtime dispatch based on IFMA availability
        if has_ifma() {
            for i in 0..15 {
                mul_256x256_ifma(
                    ws.eval_a[i].as_ptr(),
                    ws.eval_b[i].as_ptr(),
                    ws.products[i].as_mut_ptr(),
                );
            }
        } else {
            for i in 0..15 {
                mul_256x256_muludq(
                    ws.eval_a[i].as_ptr(),
                    ws.eval_b[i].as_ptr(),
                    ws.products[i].as_mut_ptr(),
                );
            }
        }
        
        // Phase 4: Interpolation
        interp_sym_antisym(ws.as_mut() as *mut _);
        // ... remaining interpolation steps ...
        
        // Phase 5: Recomposition with carry propagation
        propagate_carries(ws.as_mut() as *mut _, result as *mut _);
    }
}

// ============================================================================
// READYOUTHIS
// ============================================================================
//
// For "every cycle matters" optimization on Xeon Scalable / Xeon 6:
//
// 1 PORTS (Sapphire Rapids):
//    - VPADDQ/VPSUBQ: ports 0, 1, 5
//    - VPSLLQ/VPSRLQ: ports 0, 1
//    - VPMULUDQ: PORT 0 ONLY (!!!)
//    - VMOVDQA64: ports 2, 3 (load), port 4 (store)
//    - Interleave adds/shifts.
//    - IFMA reduces chain length by 2-3x
//
// 2 Optimization notes
//    - Byte aligned: fetch 2048[x2] bytes ahead
//    - Non-temporal stores unless reading reg
//    - Keep hot loops under 32 uop (buf)
//    - Align to 32B for uop cache efficiency
//    - Avoid crossing 64B boundaries in tight loops
//    - 512-bit frequency throttling on older Xeon Scalable mitigated with "2 256" register allocation
//    - MUST USE ZMM ops, no mixing ymm destinations
//    - Store-to-load 16-byte aligned accesses for load sub store

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_detection() {
        println!("IFMA available: {}", has_ifma());
    }
    
    #[test]
    fn test_workspace_alignment() {
        let ws = Toom8Workspace::new();
        assert_eq!(ws.as_ref() as *const _ as usize % 64, 0);
    }
}
