// Chi computation worker with private memory
// Each worker has its own WASM instance (no SharedArrayBuffer)

let wasmInstance = null;

// GF(2^128) multiplication with reduction (BearSSL/polyval algorithm)
// Matches the Rust clmul soft64 implementation
function gf128_mul(a, b) {
    // a and b are Uint8Array(16) in little-endian
    // Convert to two 64-bit BigInts
    const a0 = readU64LE(a, 0);
    const a1 = readU64LE(a, 8);
    const b0 = readU64LE(b, 0);
    const b1 = readU64LE(b, 8);

    // Carryless multiplication of 64-bit halves
    const z0 = clmul64(a0, b0);
    const z1 = clmul64(a0, b1) ^ clmul64(a1, b0);
    const z2 = clmul64(a1, b1);

    // Combine: result is z2:z1:z0 (256-bit)
    // z1 straddles the middle
    const r1 = z1.lo ^ z0.hi;
    const r2 = z1.hi ^ z2.lo;

    // Now we have 256-bit result: [z2.hi, r2, r1, z0.lo]
    // Need to reduce mod x^128 + x^127 + x^126 + x^121 + 1 (POLYVAL polynomial)

    // Reduction: fold high 128 bits into low 128 bits
    // Using the polyval reduction polynomial
    let lo = z0.lo;
    let hi = r1;
    const c0 = r2;
    const c1 = z2.hi;

    // Fold c1 (bits 192-255)
    hi ^= c1 ^ (c1 >> 1n) ^ (c1 >> 2n) ^ (c1 >> 7n);
    lo ^= (c1 << 63n) ^ (c1 << 62n) ^ (c1 << 57n);

    // Fold c0 (bits 128-191)
    hi ^= c0 ^ (c0 >> 1n) ^ (c0 >> 2n) ^ (c0 >> 7n);
    lo ^= (c0 << 63n) ^ (c0 << 62n) ^ (c0 << 57n);

    // Mask to 64 bits
    lo = lo & 0xFFFFFFFFFFFFFFFFn;
    hi = hi & 0xFFFFFFFFFFFFFFFFn;

    // Write result back to Uint8Array
    const result = new Uint8Array(16);
    writeU64LE(result, 0, lo);
    writeU64LE(result, 8, hi);
    return result;
}

// Carryless multiplication of two 64-bit values
// Returns { lo: BigInt, hi: BigInt } representing 128-bit result
function clmul64(a, b) {
    let lo = 0n;
    let hi = 0n;

    for (let i = 0n; i < 64n; i++) {
        if ((b >> i) & 1n) {
            lo ^= a << i;
            if (i > 0n) {
                hi ^= a >> (64n - i);
            }
        }
    }

    return { lo: lo & 0xFFFFFFFFFFFFFFFFn, hi };
}

// Read 64-bit little-endian BigInt from Uint8Array
function readU64LE(arr, offset) {
    let val = 0n;
    for (let i = 0; i < 8; i++) {
        val |= BigInt(arr[offset + i]) << BigInt(i * 8);
    }
    return val;
}

// Write 64-bit little-endian BigInt to Uint8Array
function writeU64LE(arr, offset, val) {
    for (let i = 0; i < 8; i++) {
        arr[offset + i] = Number((val >> BigInt(i * 8)) & 0xFFn);
    }
}

// Blake3 hash (simplified - uses SubtleCrypto SHA-256 as fallback)
// For production, should use proper blake3-js library
async function blake3Hash(data) {
    // Note: This uses SHA-256 as a placeholder. For correct operation,
    // replace with actual blake3 implementation (e.g., blake3-js package)
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    return new Uint8Array(hashBuffer);
}

// Compute chi starting points using same algorithm as Rust
// Bootstrap 16 values via squaring, hash each to get independent starts
async function computeChiStarts(chi, segmentSize) {
    const PARALLELISM = 16;
    const bootstrapped = [];

    // Bootstrap 16 values via squaring
    let current = new Uint8Array(chi);
    for (let i = 0; i < PARALLELISM; i++) {
        bootstrapped.push(new Uint8Array(current));
        current = gf128_mul(current, current);
    }

    // Hash each to get independent starting points
    const starts = [];
    for (let i = 0; i < PARALLELISM; i++) {
        const toHash = new Uint8Array(16 + 8 + 8);
        toHash.set(bootstrapped[i], 0);
        // i as u64 little-endian
        const iBytes = new Uint8Array(8);
        writeU64LE(iBytes, 0, BigInt(i));
        toHash.set(iBytes, 16);
        // segmentSize as u64 little-endian
        const segBytes = new Uint8Array(8);
        writeU64LE(segBytes, 0, BigInt(segmentSize));
        toHash.set(segBytes, 24);

        const hash = await blake3Hash(toHash);
        starts.push(hash.slice(0, 16));
    }

    return starts;
}

// Compute a segment of chi values
function computeChiSegment(start, count) {
    const result = new Uint8Array(count * 16);
    let current = new Uint8Array(start);

    for (let i = 0; i < count; i++) {
        result.set(current, i * 16);
        current = gf128_mul(current, current);
    }

    return result;
}

// Handle messages from main thread
self.onmessage = async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            // Worker is ready (no WASM needed for pure JS gfmul)
            self.postMessage({ type: 'ready' });
            break;

        case 'compute_segment':
            // Compute a segment of chi values
            const { segmentIndex, start, count, requestId } = data;
            const segment = computeChiSegment(start, count);
            self.postMessage({
                type: 'segment_result',
                segmentIndex,
                data: segment,
                requestId
            });
            break;

        case 'compute_starts':
            // Compute all starting points (done once per computation)
            const { chi, segmentSize, requestId: startsRequestId } = data;
            const starts = await computeChiStarts(chi, segmentSize);
            self.postMessage({
                type: 'starts_result',
                starts: starts.map(s => Array.from(s)),
                requestId: startsRequestId
            });
            break;
    }
};
