//! Clustered pair streams for cache-friendly, cutoff-aware force traversal.

use lin_alg::f32::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterRange {
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterPair {
    pub target: u32,
    pub source: u32,
}

/// Immutable pair stream sorted by target and source cluster.  The stream keeps
/// pair indices compact and exposes cluster ranges so workers can own blocks
/// without materialising a full `Vec<NonBondedPair>` per worker.
#[derive(Clone, Debug, Default)]
pub struct ClusterPairStream {
    pub cluster_size: usize,
    pub target: Vec<u32>,
    pub source: Vec<u32>,
    pub target_clusters: Vec<ClusterRange>,
    pub source_clusters: Vec<ClusterRange>,
}

impl ClusterPairStream {
    pub fn from_pairs(pairs: &[(u32, u32)], cluster_size: usize) -> Result<Self, String> {
        if cluster_size == 0 {
            return Err("cluster size must be non-zero".into());
        }
        let mut p = pairs.to_vec();
        p.sort_unstable_by_key(|&(t, s)| {
            (t as usize / cluster_size, s as usize / cluster_size, t, s)
        });
        let mut out = Self {
            cluster_size,
            ..Default::default()
        };
        out.target.reserve(p.len());
        out.source.reserve(p.len());
        for &(t, s) in &p {
            out.target.push(t);
            out.source.push(s);
        }
        out.target_clusters = ranges_by_cluster(&out.target, cluster_size);
        out.source_clusters = ranges_by_cluster(&out.source, cluster_size);
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.target.len()
    }
    pub fn is_empty(&self) -> bool {
        self.target.is_empty()
    }
    pub fn pair(&self, i: usize) -> Option<ClusterPair> {
        Some(ClusterPair {
            target: *self.target.get(i)?,
            source: *self.source.get(i)?,
        })
    }

    /// Build a scalar cutoff mask for a compact pair tile.  The mask is kept
    /// separate from the stream so the hot kernel can consume it as a bitset.
    pub fn cutoff_mask<F>(&self, start: usize, len: usize, mut within_cutoff: F) -> u64
    where
        F: FnMut(u32, u32) -> bool,
    {
        let end = (start + len).min(self.len()).min(start + 64);
        let mut mask = 0u64;
        for i in start..end {
            if within_cutoff(self.target[i], self.source[i]) {
                mask |= 1u64 << (i - start);
            }
        }
        mask
    }

    /// Return false when two cluster bounding boxes cannot contain a pair within cutoff.
    pub fn bbox_may_interact(
        a_min: Vec3,
        a_max: Vec3,
        b_min: Vec3,
        b_max: Vec3,
        cutoff: f32,
    ) -> bool {
        let gap = |alo: f32, ahi: f32, blo: f32, bhi: f32| {
            if ahi < blo {
                blo - ahi
            } else if bhi < alo {
                alo - bhi
            } else {
                0.0
            }
        };
        let d2 = gap(a_min.x, a_max.x, b_min.x, b_max.x).powi(2)
            + gap(a_min.y, a_max.y, b_min.y, b_max.y).powi(2)
            + gap(a_min.z, a_max.z, b_min.z, b_max.z).powi(2);
        d2 <= cutoff * cutoff
    }
}

fn ranges_by_cluster(indices: &[u32], cluster_size: usize) -> Vec<ClusterRange> {
    let max_cluster = indices.iter().map(|&i| i as usize / cluster_size).max();
    let Some(max_cluster) = max_cluster else {
        return Vec::new();
    };
    let mut ranges = vec![ClusterRange { start: 0, end: 0 }; max_cluster + 1];
    for (i, &value) in indices.iter().enumerate() {
        let c = value as usize / cluster_size;
        if ranges[c].start == ranges[c].end {
            ranges[c].start = i as u32;
        }
        ranges[c].end = i as u32 + 1;
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sorts_by_target_then_source_cluster() {
        let s = ClusterPairStream::from_pairs(&[(17, 1), (1, 17), (2, 3)], 16).unwrap();
        assert_eq!(s.target, vec![2, 1, 17]);
        assert_eq!(s.target_clusters[0], ClusterRange { start: 0, end: 2 });
    }
    #[test]
    fn cutoff_mask_marks_only_selected_lanes() {
        let s = ClusterPairStream::from_pairs(&[(0, 1), (0, 2), (16, 17)], 16).unwrap();
        let mask = s.cutoff_mask(0, 3, |target, source| target == 0 && source == 2);
        assert_eq!(mask, 0b010);
    }

    #[test]
    fn bbox_pruning_is_conservative() {
        assert!(!ClusterPairStream::bbox_may_interact(
            Vec3::new(0., 0., 0.),
            Vec3::new(1., 1., 1.),
            Vec3::new(5., 0., 0.),
            Vec3::new(6., 1., 1.),
            1.
        ));
        assert!(ClusterPairStream::bbox_may_interact(
            Vec3::new(0., 0., 0.),
            Vec3::new(1., 1., 1.),
            Vec3::new(1.5, 0., 0.),
            Vec3::new(2., 1., 1.),
            1.
        ));
    }
}
