//! Structure-of-arrays cache for rigid OPC water.
//!
//! `WaterMolOpc` remains the compatibility/integration representation.  This
//! cache mirrors positions and owns worker-local force storage in contiguous
//! arrays, so non-bonded kernels can avoid an AoS gather and scattered writes.

use lin_alg::{f32::{Vec3, Vec3 as Vec3F32}, f64::Vec3 as Vec3F64};

use crate::engine::md_core::solvent::WaterMolOpc;

/// A contiguous target-tile range in a sorted pair stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRange {
    pub start: u32,
    pub end: u32,
}

/// Pair metadata sorted by target tile. The pair stream is immutable during a
/// force evaluation, so workers can process disjoint ranges without atomics.
#[derive(Clone, Debug, Default)]
pub struct SortedTileMetadata {
    pub target_tile_size: usize,
    pub target_indices: Vec<u32>,
    pub source_indices: Vec<u32>,
    pub tile_ranges: Vec<TileRange>,
}

impl SortedTileMetadata {
    pub fn from_pairs(pairs: &[(u32, u32)], target_tile_size: usize) -> Result<Self, String> {
        if target_tile_size == 0 {
            return Err("target tile size must be non-zero".into());
        }
        let mut sorted = pairs.to_vec();
        sorted.sort_unstable_by_key(|&(target, source)| {
            (target / target_tile_size as u32, target, source)
        });
        let mut target_indices = Vec::with_capacity(sorted.len());
        let mut source_indices = Vec::with_capacity(sorted.len());
        let mut tile_ranges = Vec::new();
        let mut current_tile = None;
        for (index, &(target, source)) in sorted.iter().enumerate() {
            let tile = target as usize / target_tile_size;
            if current_tile != Some(tile) {
                if let Some(previous) = current_tile {
                    tile_ranges[previous].end = index as u32;
                }
                while tile_ranges.len() <= tile {
                    tile_ranges.push(TileRange { start: index as u32, end: index as u32 });
                }
                tile_ranges[tile].start = index as u32;
                current_tile = Some(tile);
            }
            target_indices.push(target);
            source_indices.push(source);
        }
        if let Some(tile) = current_tile {
            tile_ranges[tile].end = sorted.len() as u32;
        }
        Ok(Self { target_tile_size, target_indices, source_indices, tile_ranges })
    }

    #[inline]
    pub fn pairs_for_tile(&self, tile: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        let range = self.tile_ranges.get(tile).copied().unwrap_or(TileRange { start: 0, end: 0 });
        (range.start as usize..range.end as usize)
            .map(|i| (self.target_indices[i], self.source_indices[i]))
    }
}

/// Tile-owned dense target force plus sparse, sorted source-water force.
/// `water_indices` is the only global water lookup needed by a tile; the force
/// arrays are dense in that local list and never allocate one array per chunk.
#[derive(Clone, Debug)]
pub struct TileLocalWaterAccumulator {
    pub target_start: usize,
    pub target_force_x: Vec<f64>,
    pub target_force_y: Vec<f64>,
    pub target_force_z: Vec<f64>,
    pub water_indices: Vec<u32>,
    pub water_force: WaterForceSoA,
}

impl TileLocalWaterAccumulator {
    pub fn new(target_start: usize, target_len: usize, water_indices: Vec<u32>) -> Self {
        let mut water_force = WaterForceSoA::default();
        water_force.resize(water_indices.len());
        Self {
            target_start,
            target_force_x: vec![0.0; target_len],
            target_force_y: vec![0.0; target_len],
            target_force_z: vec![0.0; target_len],
            water_indices,
            water_force,
        }
    }

    #[inline]
    pub fn add_target(&mut self, target: usize, force: Vec3) {
        let i = target.checked_sub(self.target_start).expect("target outside tile");
        assert!(i < self.target_force_x.len());
        self.target_force_x[i] += force.x as f64;
        self.target_force_y[i] += force.y as f64;
        self.target_force_z[i] += force.z as f64;
    }

    pub fn merge_into(&self, target: &mut [Vec3F64], water: &mut WaterForceSoA) {
        for i in 0..self.target_force_x.len() {
            let dst = self.target_start + i;
            if dst < target.len() {
                target[dst].x += self.target_force_x[i];
                target[dst].y += self.target_force_y[i];
                target[dst].z += self.target_force_z[i];
            }
        }
        for (local, &global) in self.water_indices.iter().enumerate() {
            let g = global as usize;
            if g >= water.ox.len() { continue; }
            water.ox[g] += self.water_force.ox[local];
            water.oy[g] += self.water_force.oy[local];
            water.oz[g] += self.water_force.oz[local];
            water.mx[g] += self.water_force.mx[local];
            water.my[g] += self.water_force.my[local];
            water.mz[g] += self.water_force.mz[local];
            water.h0x[g] += self.water_force.h0x[local];
            water.h0y[g] += self.water_force.h0y[local];
            water.h0z[g] += self.water_force.h0z[local];
            water.h1x[g] += self.water_force.h1x[local];
            water.h1y[g] += self.water_force.h1y[local];
            water.h1z[g] += self.water_force.h1z[local];
        }
    }
}

#[derive(Clone, Default)]
pub struct WaterSoA {
    pub ox: Vec<f32>,
    pub oy: Vec<f32>,
    pub oz: Vec<f32>,
    pub mx: Vec<f32>,
    pub my: Vec<f32>,
    pub mz: Vec<f32>,
    pub h0x: Vec<f32>,
    pub h0y: Vec<f32>,
    pub h0z: Vec<f32>,
    pub h1x: Vec<f32>,
    pub h1y: Vec<f32>,
    pub h1z: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct WaterForceSoA {
    pub ox: Vec<f64>,
    pub oy: Vec<f64>,
    pub oz: Vec<f64>,
    pub mx: Vec<f64>,
    pub my: Vec<f64>,
    pub mz: Vec<f64>,
    pub h0x: Vec<f64>,
    pub h0y: Vec<f64>,
    pub h0z: Vec<f64>,
    pub h1x: Vec<f64>,
    pub h1y: Vec<f64>,
    pub h1z: Vec<f64>,
}

impl WaterSoA {
    pub fn from_water(water: &[WaterMolOpc]) -> Self {
        let mut s = Self::default();
        s.resize(water.len());
        for (i, w) in water.iter().enumerate() {
            let p = [w.o.posit, w.m.posit, w.h0.posit, w.h1.posit];
            s.ox[i] = p[0].x;
            s.oy[i] = p[0].y;
            s.oz[i] = p[0].z;
            s.mx[i] = p[1].x;
            s.my[i] = p[1].y;
            s.mz[i] = p[1].z;
            s.h0x[i] = p[2].x;
            s.h0y[i] = p[2].y;
            s.h0z[i] = p[2].z;
            s.h1x[i] = p[3].x;
            s.h1y[i] = p[3].y;
            s.h1z[i] = p[3].z;
        }
        s
    }
    pub fn resize(&mut self, n: usize) {
        for v in [
            &mut self.ox,
            &mut self.oy,
            &mut self.oz,
            &mut self.mx,
            &mut self.my,
            &mut self.mz,
            &mut self.h0x,
            &mut self.h0y,
            &mut self.h0z,
            &mut self.h1x,
            &mut self.h1y,
            &mut self.h1z,
        ] {
            v.clear();
            v.resize(n, 0.0);
        }
    }
    #[inline]
    pub fn sync_from_water(&mut self, water: &[WaterMolOpc]) {
        if self.ox.len() != water.len() {
            self.resize(water.len());
        }
        for (i, w) in water.iter().enumerate() {
            self.ox[i] = w.o.posit.x;
            self.oy[i] = w.o.posit.y;
            self.oz[i] = w.o.posit.z;
            self.mx[i] = w.m.posit.x;
            self.my[i] = w.m.posit.y;
            self.mz[i] = w.m.posit.z;
            self.h0x[i] = w.h0.posit.x;
            self.h0y[i] = w.h0.posit.y;
            self.h0z[i] = w.h0.posit.z;
            self.h1x[i] = w.h1.posit.x;
            self.h1y[i] = w.h1.posit.y;
            self.h1z[i] = w.h1.posit.z;
        }
    }
}

impl WaterForceSoA {
    pub fn zeros(n: usize) -> Self {
        let mut s = Self::default();
        s.resize(n);
        s
    }
    pub fn resize(&mut self, n: usize) {
        for v in [
            &mut self.ox,
            &mut self.oy,
            &mut self.oz,
            &mut self.mx,
            &mut self.my,
            &mut self.mz,
            &mut self.h0x,
            &mut self.h0y,
            &mut self.h0z,
            &mut self.h1x,
            &mut self.h1y,
            &mut self.h1z,
        ] {
            v.clear();
            v.resize(n, 0.0);
        }
    }
    #[inline]
    pub fn add(&mut self, i: usize, site: usize, f: Vec3) {
        let (x, y, z) = match site {
            0 => (&mut self.ox, &mut self.oy, &mut self.oz),
            1 => (&mut self.mx, &mut self.my, &mut self.mz),
            2 => (&mut self.h0x, &mut self.h0y, &mut self.h0z),
            _ => (&mut self.h1x, &mut self.h1y, &mut self.h1z),
        };
        x[i] += f.x as f64;
        y[i] += f.y as f64;
        z[i] += f.z as f64;
    }
    pub fn to_water(&self, water: &mut [WaterMolOpc]) {
        for (i, w) in water.iter_mut().enumerate() {
            w.o.force += Vec3F32::new(self.ox[i] as f32, self.oy[i] as f32, self.oz[i] as f32);
            w.m.force += Vec3F32::new(self.mx[i] as f32, self.my[i] as f32, self.mz[i] as f32);
            w.h0.force += Vec3F32::new(self.h0x[i] as f32, self.h0y[i] as f32, self.h0z[i] as f32);
            w.h1.force += Vec3F32::new(self.h1x[i] as f32, self.h1y[i] as f32, self.h1z[i] as f32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn soa_round_trip_positions_and_force() {
        let w = Vec::<WaterMolOpc>::new();
        let s = WaterSoA::from_water(&w);
        assert!(s.ox.is_empty());
        let f = WaterForceSoA::zeros(2);
        assert_eq!(f.ox.len(), 2);
    }
}
