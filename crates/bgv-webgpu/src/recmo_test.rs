//! Tests for recmo's Goldilocks WGSL components.
//! Tests each component individually to identify bugs.

use pollster::FutureExt;
use wgpu::util::DeviceExt;

/// The Goldilocks prime p = 2^64 - 2^32 + 1
const GOLDILOCKS_P: u64 = 0xFFFFFFFF00000001;

/// CPU reference: hadd - computes (a + b) / 2, used for carry detection
/// This is NOT the high 32 bits, it's (a >> 1) + (b >> 1) + ((a & b) & 1)
fn cpu_hadd(a: u32, b: u32) -> u32 {
    // (a + b) / 2 computed without overflow
    (a >> 1) + (b >> 1) + ((a & b) & 1)
}

/// CPU reference: mul64 - 32x32 -> 64-bit multiplication
fn cpu_mul64(a: u32, b: u32) -> u64 {
    a as u64 * b as u64
}

/// CPU reference: mul128 - 64x64 -> 128-bit multiplication
fn cpu_mul128(a: u64, b: u64) -> u128 {
    a as u128 * b as u128
}

/// CPU reference: Goldilocks reduction
fn cpu_reduce_goldilocks(n: u128) -> u64 {
    (n % GOLDILOCKS_P as u128) as u64
}

/// CPU reference: Goldilocks addition
fn cpu_add_goldilocks(a: u64, b: u64) -> u64 {
    let sum = a as u128 + b as u128;
    (sum % GOLDILOCKS_P as u128) as u64
}

/// CPU reference: Goldilocks subtraction
fn cpu_sub_goldilocks(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        GOLDILOCKS_P - (b - a)
    }
}

/// CPU reference: Goldilocks multiplication
fn cpu_mul_goldilocks(a: u64, b: u64) -> u64 {
    cpu_reduce_goldilocks(a as u128 * b as u128)
}

/// Helper to convert u64 to vec2<u32> (little-endian)
fn u64_to_vec2(v: u64) -> [u32; 2] {
    [v as u32, (v >> 32) as u32]
}

/// Helper to convert vec2<u32> to u64
fn vec2_to_u64(v: [u32; 2]) -> u64 {
    v[0] as u64 | ((v[1] as u64) << 32)
}

/// Helper to convert u128 to vec4<u32>
fn u128_to_vec4(v: u128) -> [u32; 4] {
    [
        v as u32,
        (v >> 32) as u32,
        (v >> 64) as u32,
        (v >> 96) as u32,
    ]
}

/// Helper to convert vec4<u32> to u128
fn vec4_to_u128(v: [u32; 4]) -> u128 {
    v[0] as u128
        | ((v[1] as u128) << 32)
        | ((v[2] as u128) << 64)
        | ((v[3] as u128) << 96)
}

/// WGSL shader for testing hadd
const TEST_HADD_SHADER: &str = r#"
// recmo's hadd function
fn hadd(a: u32, b: u32) -> u32 {
    return (a >> 1u) + (b >> 1u) + ((a & b) & 1u);
}

@group(0) @binding(0) var<storage, read_write> input_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> input_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx < arrayLength(&input_a) {
        output[idx] = hadd(input_a[idx], input_b[idx]);
    }
}
"#;

/// WGSL shader for testing mul64
const TEST_MUL64_SHADER: &str = r#"
fn hadd(a: u32, b: u32) -> u32 {
    return (a >> 1u) + (b >> 1u) + ((a & b) & 1u);
}

fn mul64(a: u32, b: u32) -> vec2<u32> {
    var a0 = (a << 16u) >> 16u;
    var a1 = a >> 16u;
    var b0 = (b << 16u) >> 16u;
    var b1 = b >> 16u;

    var a0b0 = a0 * b0;
    var a0b1 = a0 * b1;
    var a1b0 = a1 * b0;
    var a1b1 = a1 * b1;

    var r: vec2<u32>;
    r.x = a0b0 + (a1b0 << 16u) + (a0b1 << 16u);
    r.y = a1b1 + (hadd((a0b0 >> 16u) + a0b1, a1b0) >> 15u);
    return r;
}

@group(0) @binding(0) var<storage, read_write> input_a: array<u32>;
@group(0) @binding(1) var<storage, read_write> input_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx < arrayLength(&input_a) {
        output[idx] = mul64(input_a[idx], input_b[idx]);
    }
}
"#;

/// WGSL shader for testing mul128
const TEST_MUL128_SHADER: &str = r#"
fn hadd(a: u32, b: u32) -> u32 {
    return (a >> 1u) + (b >> 1u) + ((a & b) & 1u);
}

fn mul64(a: u32, b: u32) -> vec2<u32> {
    var a0 = (a << 16u) >> 16u;
    var a1 = a >> 16u;
    var b0 = (b << 16u) >> 16u;
    var b1 = b >> 16u;

    var a0b0 = a0 * b0;
    var a0b1 = a0 * b1;
    var a1b0 = a1 * b0;
    var a1b1 = a1 * b1;

    var r: vec2<u32>;
    r.x = a0b0 + (a1b0 << 16u) + (a0b1 << 16u);
    r.y = a1b1 + (hadd((a0b0 >> 16u) + a0b1, a1b0) >> 15u);
    return r;
}

fn mul128(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    var a0b0 = mul64(a.x, b.x);
    var a0b1 = mul64(a.x, b.y);
    var a1b0 = mul64(a.y, b.x);
    var a1b1 = mul64(a.y, b.y);

    var r = vec4<u32>(a0b0, a1b1);

    r.y += a0b1.x;
    if (r.y < a0b1.x) {
        a0b1.y += 1u;
    }
    r.z += a0b1.y;
    if (r.z < a0b1.y) {
        r.w += 1u;
    }

    r.y += a1b0.x;
    if (r.y < a1b0.x) {
        a1b0.y += 1u;
    }
    r.z += a1b0.y;
    if (r.z < a1b0.y) {
        r.w += 1u;
    }

    return r;
}

@group(0) @binding(0) var<storage, read_write> input_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> input_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<u32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx < arrayLength(&input_a) {
        output[idx] = mul128(input_a[idx], input_b[idx]);
    }
}
"#;

/// WGSL shader for testing Goldilocks reduce - FIXED VERSION
/// Fixed underflow handling for large n.z and n.w values
const TEST_REDUCE_SHADER: &str = r#"
// Goldilocks prime p = 2^64 - 2^32 + 1
const P_LO: u32 = 0x00000001u;
const P_HI: u32 = 0xFFFFFFFFu;
const EPSILON: u32 = 0xFFFFFFFFu; // 2^64 mod p = 2^32 - 1

// Reduce 128-bit n to 64-bit result mod p
// Formula: n.x + n.y*2^32 + n.z*2^64 + n.w*2^96 mod p
// Since 2^64 ≡ 2^32 - 1 (mod p) and 2^96 ≡ -1 (mod p):
// result ≡ (n.x - n.z - n.w) + (n.y + n.z) * 2^32 (mod p)
fn reduce(n: vec4<u32>) -> vec2<u32> {
    // Compute (n.y + n.z) with carry
    var mid = n.y + n.z;
    var mid_carry = u32(mid < n.y);

    // Compute n.z + n.w with carry (for subtraction from low part)
    var sub_total = n.z + n.w;
    var sub_carry = u32(sub_total < n.z);

    // Initialize result
    var r_lo: u32;
    var r_hi: u32;

    // Handle subtraction: n.x - (n.z + n.w)
    if (n.x >= sub_total) {
        r_lo = n.x - sub_total;
        // No borrow needed, r_hi = mid - sub_carry
        if (mid >= sub_carry) {
            r_hi = mid - sub_carry;
        } else {
            // Underflow in high part, need to add p
            r_hi = mid - sub_carry; // wraps
            // Add p: (r_lo, r_hi) + (1, 0xFFFFFFFF)
            let old_lo = r_lo;
            r_lo = r_lo + P_LO;
            if (r_lo < old_lo) {
                r_hi = r_hi + 1u;
            }
            r_hi = r_hi + P_HI;
        }
    } else {
        // n.x < sub_total, need to borrow
        r_lo = n.x - sub_total; // wraps, effectively n.x + 2^32 - sub_total
        // Borrow 1 from mid, then subtract sub_carry
        var borrow = sub_carry + 1u;
        if (mid >= borrow) {
            r_hi = mid - borrow;
        } else {
            // Underflow in high part, add p
            r_hi = mid - borrow; // wraps
            let old_lo = r_lo;
            r_lo = r_lo + P_LO;
            if (r_lo < old_lo) {
                r_hi = r_hi + 1u;
            }
            r_hi = r_hi + P_HI;
        }
    }

    // Handle mid_carry: if (n.y + n.z) overflowed, add 2^64 mod p = EPSILON
    if (mid_carry > 0u) {
        let old_lo = r_lo;
        r_lo = r_lo + EPSILON;
        if (r_lo < old_lo) {
            r_hi = r_hi + 1u;
        }
    }

    // Final reduction: while result >= p, subtract p
    // p = 0xFFFFFFFF00000001, so r >= p iff r_hi == 0xFFFFFFFF && r_lo >= 1
    for (var i = 0u; i < 3u; i = i + 1u) {
        if (r_hi == P_HI && r_lo >= P_LO) {
            var borrow = u32(r_lo < P_LO);
            r_lo = r_lo - P_LO;
            r_hi = r_hi - P_HI - borrow;
        } else {
            break;
        }
    }

    return vec2<u32>(r_lo, r_hi);
}

@group(0) @binding(0) var<storage, read_write> input: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx < arrayLength(&input) {
        output[idx] = reduce(input[idx]);
    }
}
"#;

/// WGSL shader for testing Goldilocks add - FIXED VERSION
/// The original recmo add doesn't reduce when p <= result < 2^64
const TEST_ADD_SHADER: &str = r#"
// Goldilocks prime p = 2^64 - 2^32 + 1
const P_LO: u32 = 0x00000001u;
const P_HI: u32 = 0xFFFFFFFFu;
const EPSILON: u32 = 0xFFFFFFFFu; // 2^64 mod p = 2^32 - 1

fn add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var r = a + b;
    var carry = u32(r.x < a.x);
    r.y = r.y + carry;

    // Check for overflow past 2^64
    var overflow = (r.y < a.y) || (carry > 0u && r.y == a.y && b.y == 0xFFFFFFFFu);
    if (r.y < a.y) {
        // Add (2^64 mod p) = EPSILON = 2^32 - 1
        let old_x = r.x;
        r.x = r.x + EPSILON;
        if (r.x < old_x) {
            r.y = r.y + 1u;
        }
    }

    // Reduce if r >= p (check if r.y > P_HI, or r.y == P_HI && r.x >= P_LO)
    // p = (P_LO=1, P_HI=0xFFFFFFFF)
    // Since P_HI is max u32, r.y > P_HI is impossible
    // So just check r.y == P_HI && r.x >= P_LO (i.e., r.y == 0xFFFFFFFF && r.x >= 1)
    if (r.y == P_HI && r.x >= P_LO) {
        // Subtract p: r = r - p
        // r.x = r.x - P_LO = r.x - 1
        // r.y = r.y - P_HI - borrow
        var borrow = u32(r.x < P_LO);
        r.x = r.x - P_LO;
        r.y = r.y - P_HI - borrow;
    }

    return r;
}

@group(0) @binding(0) var<storage, read_write> input_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> input_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx < arrayLength(&input_a) {
        output[idx] = add(input_a[idx], input_b[idx]);
    }
}
"#;

/// WGSL shader for testing Goldilocks sub
const TEST_SUB_SHADER: &str = r#"
fn sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    var r = a - b;
    r.y -= u32(r.x > a.x);
    if (r.y > a.y) {
        // Add 2^64 mod p
        r.x += 1u;
        r.y -= u32(r.x != 0u);
    }
    return r;
}

@group(0) @binding(0) var<storage, read_write> input_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> input_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx < arrayLength(&input_a) {
        output[idx] = sub(input_a[idx], input_b[idx]);
    }
}
"#;

/// WGSL shader for testing full Goldilocks mul - FIXED VERSION
const TEST_MUL_SHADER: &str = r#"
// Goldilocks prime p = 2^64 - 2^32 + 1
const P_LO: u32 = 0x00000001u;
const P_HI: u32 = 0xFFFFFFFFu;
const EPSILON: u32 = 0xFFFFFFFFu;

fn hadd(a: u32, b: u32) -> u32 {
    return (a >> 1u) + (b >> 1u) + ((a & b) & 1u);
}

fn mul64(a: u32, b: u32) -> vec2<u32> {
    var a0 = (a << 16u) >> 16u;
    var a1 = a >> 16u;
    var b0 = (b << 16u) >> 16u;
    var b1 = b >> 16u;

    var a0b0 = a0 * b0;
    var a0b1 = a0 * b1;
    var a1b0 = a1 * b0;
    var a1b1 = a1 * b1;

    var r: vec2<u32>;
    r.x = a0b0 + (a1b0 << 16u) + (a0b1 << 16u);
    r.y = a1b1 + (hadd((a0b0 >> 16u) + a0b1, a1b0) >> 15u);
    return r;
}

fn mul128(a: vec2<u32>, b: vec2<u32>) -> vec4<u32> {
    var a0b0 = mul64(a.x, b.x);
    var a0b1 = mul64(a.x, b.y);
    var a1b0 = mul64(a.y, b.x);
    var a1b1 = mul64(a.y, b.y);

    var r = vec4<u32>(a0b0, a1b1);

    r.y += a0b1.x;
    if (r.y < a0b1.x) {
        a0b1.y += 1u;
    }
    r.z += a0b1.y;
    if (r.z < a0b1.y) {
        r.w += 1u;
    }

    r.y += a1b0.x;
    if (r.y < a1b0.x) {
        a1b0.y += 1u;
    }
    r.z += a1b0.y;
    if (r.z < a1b0.y) {
        r.w += 1u;
    }

    return r;
}

// Fixed reduce function
fn reduce(n: vec4<u32>) -> vec2<u32> {
    var mid = n.y + n.z;
    var mid_carry = u32(mid < n.y);

    var sub_total = n.z + n.w;
    var sub_carry = u32(sub_total < n.z);

    var r_lo: u32;
    var r_hi: u32;

    if (n.x >= sub_total) {
        r_lo = n.x - sub_total;
        if (mid >= sub_carry) {
            r_hi = mid - sub_carry;
        } else {
            r_hi = mid - sub_carry;
            let old_lo = r_lo;
            r_lo = r_lo + P_LO;
            if (r_lo < old_lo) {
                r_hi = r_hi + 1u;
            }
            r_hi = r_hi + P_HI;
        }
    } else {
        r_lo = n.x - sub_total;
        var borrow = sub_carry + 1u;
        if (mid >= borrow) {
            r_hi = mid - borrow;
        } else {
            r_hi = mid - borrow;
            let old_lo = r_lo;
            r_lo = r_lo + P_LO;
            if (r_lo < old_lo) {
                r_hi = r_hi + 1u;
            }
            r_hi = r_hi + P_HI;
        }
    }

    if (mid_carry > 0u) {
        let old_lo = r_lo;
        r_lo = r_lo + EPSILON;
        if (r_lo < old_lo) {
            r_hi = r_hi + 1u;
        }
    }

    for (var i = 0u; i < 3u; i = i + 1u) {
        if (r_hi == P_HI && r_lo >= P_LO) {
            var borrow = u32(r_lo < P_LO);
            r_lo = r_lo - P_LO;
            r_hi = r_hi - P_HI - borrow;
        } else {
            break;
        }
    }

    return vec2<u32>(r_lo, r_hi);
}

fn mul(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return reduce(mul128(a, b));
}

@group(0) @binding(0) var<storage, read_write> input_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> input_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if idx < arrayLength(&input_a) {
        output[idx] = mul(input_a[idx], input_b[idx]);
    }
}
"#;

fn get_device() -> (wgpu::Device, wgpu::Queue) {
    async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find GPU adapter");

        adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device")
    }.block_on()
}

#[test]
fn test_recmo_hadd() {
    let (device, queue) = get_device();

    // Test cases: pairs of (a, b)
    let test_cases: Vec<(u32, u32)> = vec![
        (0, 0),
        (1, 1),
        (0xFFFFFFFF, 1),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x80000000, 0x80000000),
        (0x12345678, 0x87654321),
    ];

    let input_a: Vec<u32> = test_cases.iter().map(|(a, _)| *a).collect();
    let input_b: Vec<u32> = test_cases.iter().map(|(_, b)| *b).collect();
    let n = test_cases.len();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_a"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_b"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_hadd"),
        source: wgpu::ShaderSource::Wgsl(TEST_HADD_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_hadd"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &staging, 0, (n * 4) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let results: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    println!("\n=== Testing recmo hadd ===");
    let mut failures = 0;
    for (i, ((a, b), &gpu_result)) in test_cases.iter().zip(results.iter()).enumerate() {
        let expected = cpu_hadd(*a, *b);
        if gpu_result != expected {
            println!("FAIL[{}]: hadd(0x{:08X}, 0x{:08X}) = 0x{:08X}, expected 0x{:08X}",
                     i, a, b, gpu_result, expected);
            failures += 1;
        } else {
            println!("PASS[{}]: hadd(0x{:08X}, 0x{:08X}) = 0x{:08X}", i, a, b, gpu_result);
        }
    }
    assert_eq!(failures, 0, "{} hadd tests failed", failures);
}

#[test]
fn test_recmo_mul64() {
    let (device, queue) = get_device();

    let test_cases: Vec<(u32, u32)> = vec![
        (0, 0),
        (1, 1),
        (2, 3),
        (0xFFFF, 0xFFFF),
        (0xFFFFFFFF, 2),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0x12345678, 0x87654321),
        (0xDEADBEEF, 0xCAFEBABE),
    ];

    let input_a: Vec<u32> = test_cases.iter().map(|(a, _)| *a).collect();
    let input_b: Vec<u32> = test_cases.iter().map(|(_, b)| *b).collect();
    let n = test_cases.len();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_a"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_b"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_mul64"),
        source: wgpu::ShaderSource::Wgsl(TEST_MUL64_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_mul64"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &staging, 0, (n * 8) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let results: Vec<[u32; 2]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    println!("\n=== Testing recmo mul64 ===");
    let mut failures = 0;
    for (i, ((a, b), result)) in test_cases.iter().zip(results.iter()).enumerate() {
        let expected = cpu_mul64(*a, *b);
        let gpu_result = vec2_to_u64(*result);
        if gpu_result != expected {
            println!("FAIL[{}]: mul64(0x{:08X}, 0x{:08X}) = 0x{:016X}, expected 0x{:016X}",
                     i, a, b, gpu_result, expected);
            failures += 1;
        } else {
            println!("PASS[{}]: mul64(0x{:08X}, 0x{:08X}) = 0x{:016X}", i, a, b, gpu_result);
        }
    }
    assert_eq!(failures, 0, "{} mul64 tests failed", failures);
}

#[test]
fn test_recmo_mul128() {
    let (device, queue) = get_device();

    let test_cases: Vec<(u64, u64)> = vec![
        (0, 0),
        (1, 1),
        (2, 3),
        (0xFFFFFFFF, 0xFFFFFFFF),
        (0xFFFFFFFFFFFFFFFF, 2),
        (0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF),
        (0x123456789ABCDEF0, 0x0FEDCBA987654321),
        (GOLDILOCKS_P - 1, GOLDILOCKS_P - 1),
    ];

    let input_a: Vec<[u32; 2]> = test_cases.iter().map(|(a, _)| u64_to_vec2(*a)).collect();
    let input_b: Vec<[u32; 2]> = test_cases.iter().map(|(_, b)| u64_to_vec2(*b)).collect();
    let n = test_cases.len();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_a"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_b"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (n * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_mul128"),
        source: wgpu::ShaderSource::Wgsl(TEST_MUL128_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_mul128"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 16) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &staging, 0, (n * 16) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let results: Vec<[u32; 4]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    println!("\n=== Testing recmo mul128 ===");
    let mut failures = 0;
    for (i, ((a, b), result)) in test_cases.iter().zip(results.iter()).enumerate() {
        let expected = cpu_mul128(*a, *b);
        let gpu_result = vec4_to_u128(*result);
        if gpu_result != expected {
            println!("FAIL[{}]: mul128(0x{:016X}, 0x{:016X})", i, a, b);
            println!("         got      0x{:032X}", gpu_result);
            println!("         expected 0x{:032X}", expected);
            failures += 1;
        } else {
            println!("PASS[{}]: mul128(0x{:016X}, 0x{:016X}) = 0x{:032X}", i, a, b, gpu_result);
        }
    }
    assert_eq!(failures, 0, "{} mul128 tests failed", failures);
}

#[test]
fn test_recmo_reduce() {
    let (device, queue) = get_device();

    // Test 128-bit values to reduce
    let test_cases: Vec<u128> = vec![
        0,
        1,
        GOLDILOCKS_P as u128 - 1,
        GOLDILOCKS_P as u128,
        GOLDILOCKS_P as u128 + 1,
        GOLDILOCKS_P as u128 * 2,
        (GOLDILOCKS_P as u128 - 1) * (GOLDILOCKS_P as u128 - 1),
        0xFFFFFFFFFFFFFFFF_FFFFFFFFFFFFFFFF_u128,
        // Products that might hit edge cases
        0x00000001_00000000_00000001_00000000,
        0xFFFFFFFF_00000000_00000000_00000001,
    ];

    let input: Vec<[u32; 4]> = test_cases.iter().map(|x| u128_to_vec4(*x)).collect();
    let n = test_cases.len();

    let buf_in = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_reduce"),
        source: wgpu::ShaderSource::Wgsl(TEST_REDUCE_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_reduce"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_in.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_out.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &staging, 0, (n * 8) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let results: Vec<[u32; 2]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    println!("\n=== Testing recmo reduce ===");
    let mut failures = 0;
    for (i, (input_val, result)) in test_cases.iter().zip(results.iter()).enumerate() {
        let expected = cpu_reduce_goldilocks(*input_val);
        let gpu_result = vec2_to_u64(*result);
        if gpu_result != expected {
            println!("FAIL[{}]: reduce(0x{:032X})", i, input_val);
            println!("         got      0x{:016X}", gpu_result);
            println!("         expected 0x{:016X}", expected);
            println!("         diff     {}", gpu_result as i128 - expected as i128);
            failures += 1;
        } else {
            println!("PASS[{}]: reduce(0x{:032X}) = 0x{:016X}", i, input_val, gpu_result);
        }
    }
    assert_eq!(failures, 0, "{} reduce tests failed", failures);
}

#[test]
fn test_recmo_add() {
    let (device, queue) = get_device();

    let test_cases: Vec<(u64, u64)> = vec![
        (0, 0),
        (1, 1),
        (GOLDILOCKS_P - 1, 1),  // Should wrap to 0
        (GOLDILOCKS_P - 1, 2),  // Should wrap to 1
        (GOLDILOCKS_P - 1, GOLDILOCKS_P - 1),  // Maximum sum
        (0xFFFFFFFF00000000, 1),  // Near high bits
        (0x8000000000000000, 0x8000000000000000),  // Overflow case
        (0xFFFFFFFFFFFFFFFF, 1),  // Maximum u64 + 1
    ];

    let input_a: Vec<[u32; 2]> = test_cases.iter().map(|(a, _)| u64_to_vec2(*a)).collect();
    let input_b: Vec<[u32; 2]> = test_cases.iter().map(|(_, b)| u64_to_vec2(*b)).collect();
    let n = test_cases.len();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_a"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_b"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_add"),
        source: wgpu::ShaderSource::Wgsl(TEST_ADD_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_add"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &staging, 0, (n * 8) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let results: Vec<[u32; 2]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    println!("\n=== Testing recmo add ===");
    let mut failures = 0;
    for (i, ((a, b), result)) in test_cases.iter().zip(results.iter()).enumerate() {
        let expected = cpu_add_goldilocks(*a, *b);
        let gpu_result = vec2_to_u64(*result);
        if gpu_result != expected {
            println!("FAIL[{}]: add(0x{:016X}, 0x{:016X})", i, a, b);
            println!("         got      0x{:016X}", gpu_result);
            println!("         expected 0x{:016X}", expected);
            println!("         diff     {} (0x{:X})", gpu_result as i128 - expected as i128,
                     (gpu_result as i128 - expected as i128).unsigned_abs());
            failures += 1;
        } else {
            println!("PASS[{}]: add(0x{:016X}, 0x{:016X}) = 0x{:016X}", i, a, b, gpu_result);
        }
    }
    assert_eq!(failures, 0, "{} add tests failed", failures);
}

#[test]
fn test_recmo_sub() {
    let (device, queue) = get_device();

    let test_cases: Vec<(u64, u64)> = vec![
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),  // Should wrap to p - 1
        (10, 5),
        (5, 10),  // Should wrap
        (GOLDILOCKS_P - 1, GOLDILOCKS_P - 1),
        (0, GOLDILOCKS_P - 1),  // 0 - (p-1) = 1
    ];

    let input_a: Vec<[u32; 2]> = test_cases.iter().map(|(a, _)| u64_to_vec2(*a)).collect();
    let input_b: Vec<[u32; 2]> = test_cases.iter().map(|(_, b)| u64_to_vec2(*b)).collect();
    let n = test_cases.len();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_a"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_b"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_sub"),
        source: wgpu::ShaderSource::Wgsl(TEST_SUB_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_sub"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &staging, 0, (n * 8) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let results: Vec<[u32; 2]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    println!("\n=== Testing recmo sub ===");
    let mut failures = 0;
    for (i, ((a, b), result)) in test_cases.iter().zip(results.iter()).enumerate() {
        let expected = cpu_sub_goldilocks(*a, *b);
        let gpu_result = vec2_to_u64(*result);
        if gpu_result != expected {
            println!("FAIL[{}]: sub(0x{:016X}, 0x{:016X})", i, a, b);
            println!("         got      0x{:016X}", gpu_result);
            println!("         expected 0x{:016X}", expected);
            println!("         diff     {} (0x{:X})", gpu_result as i128 - expected as i128,
                     (gpu_result as i128 - expected as i128).unsigned_abs());
            failures += 1;
        } else {
            println!("PASS[{}]: sub(0x{:016X}, 0x{:016X}) = 0x{:016X}", i, a, b, gpu_result);
        }
    }
    assert_eq!(failures, 0, "{} sub tests failed", failures);
}

#[test]
fn test_recmo_mul() {
    let (device, queue) = get_device();

    let test_cases: Vec<(u64, u64)> = vec![
        (0, 0),
        (1, 1),
        (2, 3),
        (1000, 1000),
        (GOLDILOCKS_P - 1, 1),
        (GOLDILOCKS_P - 1, 2),
        (GOLDILOCKS_P - 1, GOLDILOCKS_P - 1),
        (0x123456789ABCDEF0 % GOLDILOCKS_P, 0x0FEDCBA987654321 % GOLDILOCKS_P),
    ];

    let input_a: Vec<[u32; 2]> = test_cases.iter().map(|(a, _)| u64_to_vec2(*a)).collect();
    let input_b: Vec<[u32; 2]> = test_cases.iter().map(|(_, b)| u64_to_vec2(*b)).collect();
    let n = test_cases.len();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_a"),
        contents: bytemuck::cast_slice(&input_a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_b"),
        contents: bytemuck::cast_slice(&input_b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test_mul"),
        source: wgpu::ShaderSource::Wgsl(TEST_MUL_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test_mul"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_out.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: (n * 8) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &staging, 0, (n * 8) as u64);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);

    let data = slice.get_mapped_range();
    let results: Vec<[u32; 2]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();

    println!("\n=== Testing recmo mul ===");
    let mut failures = 0;
    for (i, ((a, b), result)) in test_cases.iter().zip(results.iter()).enumerate() {
        let expected = cpu_mul_goldilocks(*a, *b);
        let gpu_result = vec2_to_u64(*result);
        if gpu_result != expected {
            println!("FAIL[{}]: mul(0x{:016X}, 0x{:016X})", i, a, b);
            println!("         got      0x{:016X}", gpu_result);
            println!("         expected 0x{:016X}", expected);
            println!("         diff     {} (0x{:X})", gpu_result as i128 - expected as i128,
                     (gpu_result as i128 - expected as i128).unsigned_abs());
            failures += 1;
        } else {
            println!("PASS[{}]: mul(0x{:016X}, 0x{:016X}) = 0x{:016X}", i, a, b, gpu_result);
        }
    }
    assert_eq!(failures, 0, "{} mul tests failed", failures);
}
