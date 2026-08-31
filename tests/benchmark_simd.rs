//! CPU SIMD microbenchmark, separate from system build/solvent/PME costs.
//!
//! Run with:
//!   cargo test --release --test benchmark_simd -- --nocapture

use std::time::Instant;

#[cfg(target_arch = "x86_64")]
use spice_engine::PairBatch16;
use spice_engine::{PairBatch8, SimdBackend};
use wide::f32x8;
#[cfg(target_arch = "x86_64")]
use wide::f32x16;

#[test]
fn benchmark_pair_kernel() {
    const BATCHES: usize = 1_000_000;
    let backend = SimdBackend::detect();
    let batch = PairBatch8 {
        dx: f32x8::new([1.7, 2.0, 2.4, 3.1, 4.0, 5.5, 7.0, 9.0]),
        dy: f32x8::new([0.2, -0.3, 0.4, 0.1, -0.2, 0.5, -0.6, 0.7]),
        dz: f32x8::new([0.1, 0.4, -0.2, 0.3, 0.6, -0.1, 0.2, -0.4]),
        sigma: f32x8::splat(1.1),
        epsilon: f32x8::splat(0.2),
        charge_product: f32x8::splat(0.15),
    };
    let start = Instant::now();
    let mut checksum_simd = 0.0f64;
    for _ in 0..BATCHES {
        let result = batch.eval_lj_coulomb(332.0522);
        checksum_simd += result
            .energy
            .to_array()
            .iter()
            .map(|&v| v as f64)
            .sum::<f64>();
        std::hint::black_box(result);
    }
    let simd_seconds = start.elapsed().as_secs_f64();

    let scalar_start = Instant::now();
    let mut checksum_scalar = 0.0f64;
    for _ in 0..BATCHES {
        for lane in 0..8 {
            let (_, energy) = batch.lane_with_safety(lane, 332.0522);
            checksum_scalar += energy as f64;
        }
    }
    let scalar_seconds = scalar_start.elapsed().as_secs_f64();

    #[cfg(target_arch = "x86_64")]
    let batch16 = PairBatch16 {
        dx: f32x16::splat(2.0),
        dy: f32x16::splat(0.1),
        dz: f32x16::splat(-0.2),
        sigma: f32x16::splat(1.1),
        epsilon: f32x16::splat(0.2),
        charge_product: f32x16::splat(0.15),
    };
    #[cfg(target_arch = "x86_64")]
    let (seconds16, checksum16) = {
        let start16 = Instant::now();
        let mut checksum16 = 0.0f64;
        for _ in 0..BATCHES {
            let result = batch16.eval_lj_coulomb(332.0522);
            checksum16 += result
                .energy
                .to_array()
                .iter()
                .map(|&v| v as f64)
                .sum::<f64>();
            std::hint::black_box(result);
        }
        (start16.elapsed().as_secs_f64(), checksum16)
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (seconds16, checksum16) = (f64::NAN, f64::NAN);
    let checksum_error = (checksum_simd - checksum_scalar).abs() / checksum_scalar.abs().max(1.0);
    println!(
        "backend={backend:?} batches={BATCHES} lanes=8 simd_seconds={simd_seconds:.6} scalar_seconds={scalar_seconds:.6} simd16_seconds={seconds16:.6} simd_batches_per_sec={:.3} speedup_vs_scalar={:.3} simd16_batches_per_sec={:.3} checksum_simd={checksum_simd:.3} checksum_scalar={checksum_scalar:.3} checksum_simd16={checksum16:.3}",
        BATCHES as f64 / simd_seconds,
        scalar_seconds / simd_seconds,
        if seconds16.is_finite() {
            BATCHES as f64 / seconds16
        } else {
            f64::NAN
        },
    );
    assert!(checksum_simd.is_finite() && checksum_scalar.is_finite());
    assert!(
        checksum_error < 0.1,
        "SIMD/scalar checksum mismatch: relative error {checksum_error}"
    );
}
