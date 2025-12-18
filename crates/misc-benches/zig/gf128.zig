// GF(2^128) multiplication with reduction - extracted from Zig stdlib
// Compiled directly to WASM for benchmarking comparison

// Software carryless multiplication of two 32-bit integers.
fn clmulSoft32(x: u32, y: u32) u64 {
    const a0: u64 = x & 0x11111111;
    const a1: u64 = x & 0x22222222;
    const a2: u64 = x & 0x44444444;
    const a3: u64 = x & 0x88888888;
    const b0: u64 = y & 0x11111111;
    const b1: u64 = y & 0x22222222;
    const b2: u64 = y & 0x44444444;
    const b3: u64 = y & 0x88888888;
    const c0 = (a0 * b0) ^ (a1 * b3) ^ (a2 * b2) ^ (a3 * b1);
    const c1 = (a0 * b1) ^ (a1 * b0) ^ (a2 * b3) ^ (a3 * b2);
    const c2 = (a0 * b2) ^ (a1 * b1) ^ (a2 * b0) ^ (a3 * b3);
    const c3 = (a0 * b3) ^ (a1 * b2) ^ (a2 * b1) ^ (a3 * b0);
    return (c0 & 0x1111111111111111) | (c1 & 0x2222222222222222) | (c2 & 0x4444444444444444) | (c3 & 0x8888888888888888);
}

const Selector = enum { lo, hi, hi_lo };

// Software carryless multiplication of two 128-bit integers using 64-bit registers.
// This is the version Zig uses for wasm32/wasm64.
fn clmulSoft128_64(x_: u128, y_: u128, comptime half: Selector) u128 {
    const a: u64 = @truncate(if (half == .hi or half == .hi_lo) x_ >> 64 else x_);
    const b: u64 = @truncate(if (half == .hi) y_ >> 64 else y_);
    const a0: u32 = @truncate(a);
    const a1: u32 = @truncate(a >> 32);
    const b0: u32 = @truncate(b);
    const b1: u32 = @truncate(b >> 32);
    const lo = clmulSoft32(a0, b0);
    const hi = clmulSoft32(a1, b1);
    const mid = clmulSoft32(a0 ^ a1, b0 ^ b1) ^ lo ^ hi;
    const res_lo = lo ^ (mid << 32);
    const res_hi = hi ^ (mid >> 32);
    return @as(u128, res_lo) | (@as(u128, res_hi) << 64);
}

const I256 = struct {
    hi: u128,
    lo: u128,
    mid: u128,
};

// Multiply two 128-bit integers in GF(2^128) - schoolbook method.
fn clmul128(x: u128, y: u128) I256 {
    return .{
        .hi = clmulSoft128_64(x, y, .hi),
        .lo = clmulSoft128_64(x, y, .lo),
        .mid = clmulSoft128_64(x, y, .hi_lo) ^ clmulSoft128_64(y, x, .hi_lo),
    };
}

// Reduce a 256-bit polynomial modulo x^128 + x^127 + x^126 + x^121 + 1.
// Uses Shay Gueron's optimization.
fn reduce(x: I256) u128 {
    const hi = x.hi ^ (x.mid >> 64);
    const lo = x.lo ^ (x.mid << 64);
    const p64: u128 = ((@as(u128, 1) << 121) | (@as(u128, 1) << 126) | (@as(u128, 1) << 127)) >> 64;
    const a = clmulSoft128_64(lo, p64, .lo);
    const b = ((lo << 64) | (lo >> 64)) ^ a;
    const c = clmulSoft128_64(b, p64, .lo);
    const d = ((b << 64) | (b >> 64)) ^ c;
    return d ^ hi;
}

// GF(2^128) multiplication with reduction.
export fn gf128_mul(x: u128, y: u128) u128 {
    return reduce(clmul128(x, y));
}

// Benchmark: squaring chain - returns final value
// Performs n sequential squarings: a -> a^2 -> a^4 -> ...
export fn gf128_bench(n: u32) u128 {
    var a: u128 = 0x123456789abcdef0fedcba9876543210;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        a = reduce(clmul128(a, a));
    }
    return a;
}
