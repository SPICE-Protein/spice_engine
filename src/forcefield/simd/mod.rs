//! Portable CPU SIMD backend selection for SE.
//!
//! This module only selects an execution capability. The numerical kernels
//! remain separate so scalar and architecture-specific implementations can be
//! compared bit-for-bit within an explicit tolerance.

pub mod pair;
pub mod pair16;

pub use pair::{PairBatch8, PairResult8};
#[cfg(target_arch = "x86_64")]
pub use pair16::{PairBatch16, PairResult16};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SimdBackend {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl SimdBackend {
    #[cfg_attr(target_arch = "aarch64", allow(unreachable_code))]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            Self::Scalar
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON/ASIMD is mandatory for AArch64 userland targets.
            Self::Neon
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::Scalar
        }
    }

    pub const fn is_simd(self) -> bool {
        !matches!(self, Self::Scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detected_backend_is_valid_for_target() {
        let backend = SimdBackend::detect();
        #[cfg(target_arch = "aarch64")]
        assert_eq!(backend, SimdBackend::Neon);
        #[cfg(not(target_arch = "aarch64"))]
        let _ = backend;
    }
}
