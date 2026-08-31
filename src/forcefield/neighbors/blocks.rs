//! Target-owned atom blocks for race-free parallel force accumulation.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomBlock {
    pub start: usize,
    pub end: usize,
}

impl AtomBlock {
    pub fn new(start: usize, end: usize) -> Result<Self, String> {
        if start >= end {
            return Err(format!("invalid atom block [{start}, {end})"));
        }
        Ok(Self { start, end })
    }

    pub fn contains(self, atom: usize) -> bool {
        self.start <= atom && atom < self.end
    }
}

pub fn make_atom_blocks(atom_count: usize, block_size: usize) -> Result<Vec<AtomBlock>, String> {
    if block_size == 0 {
        return Err("atom block size must be non-zero".into());
    }
    Ok((0..atom_count)
        .step_by(block_size)
        .map(|start| AtomBlock {
            start,
            end: (start + block_size).min(atom_count),
        })
        .collect())
}

pub fn atom_block_index(atom: usize, block_size: usize) -> Result<usize, String> {
    if block_size == 0 {
        return Err("atom block size must be non-zero".into());
    }
    Ok(atom / block_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_cover_atoms_without_overlap() {
        let blocks = make_atom_blocks(10, 4).unwrap();
        assert_eq!(
            blocks,
            vec![
                AtomBlock { start: 0, end: 4 },
                AtomBlock { start: 4, end: 8 },
                AtomBlock { start: 8, end: 10 },
            ]
        );
        for atom in 0..10 {
            assert_eq!(blocks.iter().filter(|b| b.contains(atom)).count(), 1);
        }
    }
}
