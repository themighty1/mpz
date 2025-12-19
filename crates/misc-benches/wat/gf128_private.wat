;; GF(2^128) multiplication with private memory for thread-local computation
;; This module uses a private (non-shared) memory for all intermediate computation

(module
  ;; Private memory - NOT shared, each instance gets its own
  ;; This avoids SharedArrayBuffer synchronization overhead
  (memory $private 1)

  ;; Export memory for debugging if needed
  (export "private_memory" (memory $private))

  ;; ============================================================
  ;; rev64: Bit-reverse a 64-bit integer
  ;; ============================================================
  (func $rev64 (param $x i64) (result i64)
    (local $tmp i64)

    ;; x = ((x & 0x5555...) << 1) | ((x >> 1) & 0x5555...)
    (local.set $tmp
      (i64.or
        (i64.shl
          (i64.and (local.get $x) (i64.const 0x5555555555555555))
          (i64.const 1))
        (i64.and
          (i64.shr_u (local.get $x) (i64.const 1))
          (i64.const 0x5555555555555555))))
    (local.set $x (local.get $tmp))

    ;; x = ((x & 0x3333...) << 2) | ((x >> 2) & 0x3333...)
    (local.set $tmp
      (i64.or
        (i64.shl
          (i64.and (local.get $x) (i64.const 0x3333333333333333))
          (i64.const 2))
        (i64.and
          (i64.shr_u (local.get $x) (i64.const 2))
          (i64.const 0x3333333333333333))))
    (local.set $x (local.get $tmp))

    ;; x = ((x & 0x0f0f...) << 4) | ((x >> 4) & 0x0f0f...)
    (local.set $tmp
      (i64.or
        (i64.shl
          (i64.and (local.get $x) (i64.const 0x0f0f0f0f0f0f0f0f))
          (i64.const 4))
        (i64.and
          (i64.shr_u (local.get $x) (i64.const 4))
          (i64.const 0x0f0f0f0f0f0f0f0f))))
    (local.set $x (local.get $tmp))

    ;; x = ((x & 0x00ff...) << 8) | ((x >> 8) & 0x00ff...)
    (local.set $tmp
      (i64.or
        (i64.shl
          (i64.and (local.get $x) (i64.const 0x00ff00ff00ff00ff))
          (i64.const 8))
        (i64.and
          (i64.shr_u (local.get $x) (i64.const 8))
          (i64.const 0x00ff00ff00ff00ff))))
    (local.set $x (local.get $tmp))

    ;; x = ((x & 0x0000ffff...) << 16) | ((x >> 16) & 0x0000ffff...)
    (local.set $tmp
      (i64.or
        (i64.shl
          (i64.and (local.get $x) (i64.const 0x0000ffff0000ffff))
          (i64.const 16))
        (i64.and
          (i64.shr_u (local.get $x) (i64.const 16))
          (i64.const 0x0000ffff0000ffff))))
    (local.set $x (local.get $tmp))

    ;; x.rotate_right(32)
    (i64.rotr (local.get $x) (i64.const 32))
  )

  ;; ============================================================
  ;; bmul64: Carryless multiply of two 64-bit integers
  ;; Uses 4-bit interleaving technique from BearSSL
  ;; ============================================================
  (func $bmul64 (param $x i64) (param $y i64) (result i64)
    (local $x0 i64) (local $x1 i64) (local $x2 i64) (local $x3 i64)
    (local $y0 i64) (local $y1 i64) (local $y2 i64) (local $y3 i64)
    (local $z0 i64) (local $z1 i64) (local $z2 i64) (local $z3 i64)

    ;; Extract 4-bit interleaved components of x
    (local.set $x0 (i64.and (local.get $x) (i64.const 0x1111111111111111)))
    (local.set $x1 (i64.and (local.get $x) (i64.const 0x2222222222222222)))
    (local.set $x2 (i64.and (local.get $x) (i64.const 0x4444444444444444)))
    (local.set $x3 (i64.and (local.get $x) (i64.const 0x8888888888888888)))

    ;; Extract 4-bit interleaved components of y
    (local.set $y0 (i64.and (local.get $y) (i64.const 0x1111111111111111)))
    (local.set $y1 (i64.and (local.get $y) (i64.const 0x2222222222222222)))
    (local.set $y2 (i64.and (local.get $y) (i64.const 0x4444444444444444)))
    (local.set $y3 (i64.and (local.get $y) (i64.const 0x8888888888888888)))

    ;; z0 = (x0*y0) ^ (x1*y3) ^ (x2*y2) ^ (x3*y1)
    (local.set $z0
      (i64.xor
        (i64.xor
          (i64.mul (local.get $x0) (local.get $y0))
          (i64.mul (local.get $x1) (local.get $y3)))
        (i64.xor
          (i64.mul (local.get $x2) (local.get $y2))
          (i64.mul (local.get $x3) (local.get $y1)))))

    ;; z1 = (x0*y1) ^ (x1*y0) ^ (x2*y3) ^ (x3*y2)
    (local.set $z1
      (i64.xor
        (i64.xor
          (i64.mul (local.get $x0) (local.get $y1))
          (i64.mul (local.get $x1) (local.get $y0)))
        (i64.xor
          (i64.mul (local.get $x2) (local.get $y3))
          (i64.mul (local.get $x3) (local.get $y2)))))

    ;; z2 = (x0*y2) ^ (x1*y1) ^ (x2*y0) ^ (x3*y3)
    (local.set $z2
      (i64.xor
        (i64.xor
          (i64.mul (local.get $x0) (local.get $y2))
          (i64.mul (local.get $x1) (local.get $y1)))
        (i64.xor
          (i64.mul (local.get $x2) (local.get $y0))
          (i64.mul (local.get $x3) (local.get $y3)))))

    ;; z3 = (x0*y3) ^ (x1*y2) ^ (x2*y1) ^ (x3*y0)
    (local.set $z3
      (i64.xor
        (i64.xor
          (i64.mul (local.get $x0) (local.get $y3))
          (i64.mul (local.get $x1) (local.get $y2)))
        (i64.xor
          (i64.mul (local.get $x2) (local.get $y1))
          (i64.mul (local.get $x3) (local.get $y0)))))

    ;; Mask and combine
    (local.set $z0 (i64.and (local.get $z0) (i64.const 0x1111111111111111)))
    (local.set $z1 (i64.and (local.get $z1) (i64.const 0x2222222222222222)))
    (local.set $z2 (i64.and (local.get $z2) (i64.const 0x4444444444444444)))
    (local.set $z3 (i64.and (local.get $z3) (i64.const 0x8888888888888888)))

    (i64.or
      (i64.or (local.get $z0) (local.get $z1))
      (i64.or (local.get $z2) (local.get $z3)))
  )

  ;; ============================================================
  ;; gf128_mul: Full GF(2^128) multiplication with reduction
  ;; Input: two 128-bit values as (lo, hi) pairs
  ;; Output: 128-bit result as (lo, hi) pair
  ;; Uses multi-value return
  ;; ============================================================
  (func $gf128_mul (export "gf128_mul")
    (param $a_lo i64) (param $a_hi i64)
    (param $b_lo i64) (param $b_hi i64)
    (result i64 i64)

    (local $h0 i64) (local $h1 i64) (local $h0r i64) (local $h1r i64)
    (local $h2 i64) (local $h2r i64)
    (local $y0 i64) (local $y1 i64) (local $y0r i64) (local $y1r i64)
    (local $y2 i64) (local $y2r i64)
    (local $z0 i64) (local $z1 i64) (local $z2 i64)
    (local $z0h i64) (local $z1h i64) (local $z2h i64)
    (local $v0 i64) (local $v1 i64) (local $v2 i64) (local $v3 i64)

    ;; h0, h1 = a
    (local.set $h0 (local.get $a_lo))
    (local.set $h1 (local.get $a_hi))
    (local.set $h0r (call $rev64 (local.get $h0)))
    (local.set $h1r (call $rev64 (local.get $h1)))
    (local.set $h2 (i64.xor (local.get $h0) (local.get $h1)))
    (local.set $h2r (i64.xor (local.get $h0r) (local.get $h1r)))

    ;; y0, y1 = b
    (local.set $y0 (local.get $b_lo))
    (local.set $y1 (local.get $b_hi))
    (local.set $y0r (call $rev64 (local.get $y0)))
    (local.set $y1r (call $rev64 (local.get $y1)))
    (local.set $y2 (i64.xor (local.get $y0) (local.get $y1)))
    (local.set $y2r (i64.xor (local.get $y0r) (local.get $y1r)))

    ;; Compute products using bmul64
    (local.set $z0 (call $bmul64 (local.get $y0) (local.get $h0)))
    (local.set $z1 (call $bmul64 (local.get $y1) (local.get $h1)))
    (local.set $z2 (call $bmul64 (local.get $y2) (local.get $h2)))
    (local.set $z0h (call $bmul64 (local.get $y0r) (local.get $h0r)))
    (local.set $z1h (call $bmul64 (local.get $y1r) (local.get $h1r)))
    (local.set $z2h (call $bmul64 (local.get $y2r) (local.get $h2r)))

    ;; Combine
    (local.set $z2 (i64.xor (local.get $z2) (i64.xor (local.get $z0) (local.get $z1))))
    (local.set $z2h (i64.xor (local.get $z2h) (i64.xor (local.get $z0h) (local.get $z1h))))

    (local.set $z0h (i64.shr_u (call $rev64 (local.get $z0h)) (i64.const 1)))
    (local.set $z1h (i64.shr_u (call $rev64 (local.get $z1h)) (i64.const 1)))
    (local.set $z2h (i64.shr_u (call $rev64 (local.get $z2h)) (i64.const 1)))

    ;; Build 256-bit result (v0, v1, v2, v3)
    (local.set $v0 (local.get $z0))
    (local.set $v1 (i64.xor (local.get $z0h) (local.get $z2)))
    (local.set $v2 (i64.xor (local.get $z1) (local.get $z2h)))
    (local.set $v3 (local.get $z1h))

    ;; Reduction (POLYVAL polynomial)
    ;; v2 ^= v0 ^ (v0 >> 1) ^ (v0 >> 2) ^ (v0 >> 7)
    (local.set $v2
      (i64.xor (local.get $v2)
        (i64.xor
          (i64.xor (local.get $v0) (i64.shr_u (local.get $v0) (i64.const 1)))
          (i64.xor (i64.shr_u (local.get $v0) (i64.const 2))
                   (i64.shr_u (local.get $v0) (i64.const 7))))))

    ;; v1 ^= (v0 << 63) ^ (v0 << 62) ^ (v0 << 57)
    (local.set $v1
      (i64.xor (local.get $v1)
        (i64.xor
          (i64.xor (i64.shl (local.get $v0) (i64.const 63))
                   (i64.shl (local.get $v0) (i64.const 62)))
          (i64.shl (local.get $v0) (i64.const 57)))))

    ;; v3 ^= v1 ^ (v1 >> 1) ^ (v1 >> 2) ^ (v1 >> 7)
    (local.set $v3
      (i64.xor (local.get $v3)
        (i64.xor
          (i64.xor (local.get $v1) (i64.shr_u (local.get $v1) (i64.const 1)))
          (i64.xor (i64.shr_u (local.get $v1) (i64.const 2))
                   (i64.shr_u (local.get $v1) (i64.const 7))))))

    ;; v2 ^= (v1 << 63) ^ (v1 << 62) ^ (v1 << 57)
    (local.set $v2
      (i64.xor (local.get $v2)
        (i64.xor
          (i64.xor (i64.shl (local.get $v1) (i64.const 63))
                   (i64.shl (local.get $v1) (i64.const 62)))
          (i64.shl (local.get $v1) (i64.const 57)))))

    ;; Return (v2, v3) as the reduced 128-bit result
    (local.get $v2)
    (local.get $v3)
  )

  ;; ============================================================
  ;; gf128_bench: Benchmark function - chain of multiplications
  ;; Returns result to prevent optimization
  ;; ============================================================
  (func $gf128_bench (export "gf128_bench")
    (param $n i32)
    (result i64 i64)

    (local $lo i64) (local $hi i64)
    (local $i i32)
    (local $new_lo i64) (local $new_hi i64)

    ;; Initialize with test value
    (local.set $lo (i64.const 0xfedcba9876543210))
    (local.set $hi (i64.const 0x123456789abcdef0))

    ;; Loop n times, squaring each iteration
    (block $done
      (loop $loop
        (br_if $done (i32.ge_u (local.get $i) (local.get $n)))

        ;; (lo, hi) = gf128_mul((lo, hi), (lo, hi))
        (call $gf128_mul
          (local.get $lo) (local.get $hi)
          (local.get $lo) (local.get $hi))
        (local.set $hi)
        (local.set $lo)

        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )

    (local.get $lo)
    (local.get $hi)
  )
)
