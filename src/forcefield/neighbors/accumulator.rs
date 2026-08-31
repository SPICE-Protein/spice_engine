//! Force accumulators for race-free block-owned parallel evaluation.

use lin_alg::f32::Vec3;

use super::blocks::AtomBlock;

#[derive(Clone, Debug)]
pub struct BlockForceAccumulator {
    pub block: AtomBlock,
    pub forces: Vec<Vec3>,
}

impl BlockForceAccumulator {
    pub fn zeros(block: AtomBlock) -> Self {
        Self {
            forces: vec![Vec3::new_zero(); block.end - block.start],
            block,
        }
    }

    pub fn add_target(&mut self, atom: usize, force: Vec3) -> Result<(), String> {
        if !self.block.contains(atom) {
            return Err(format!("atom {atom} is outside block {:?}", self.block));
        }
        self.forces[atom - self.block.start] += force;
        Ok(())
    }

    pub fn get(&self, atom: usize) -> Option<Vec3> {
        self.block
            .contains(atom)
            .then(|| self.forces[atom - self.block.start])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulator_owns_only_its_target_block() {
        let block = AtomBlock::new(4, 8).unwrap();
        let mut acc = BlockForceAccumulator::zeros(block);
        acc.add_target(4, Vec3::new(1.0, 0.0, 0.0)).unwrap();
        acc.add_target(7, Vec3::new(0.0, 2.0, 0.0)).unwrap();
        assert_eq!(acc.get(4).unwrap().x, 1.0);
        assert_eq!(acc.get(7).unwrap().y, 2.0);
        assert!(acc.add_target(3, Vec3::new_zero()).is_err());
    }
}
