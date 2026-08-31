//! Pair partitioning for target-owned, race-free parallel force evaluation.

use super::blocks::{AtomBlock, atom_block_index, make_atom_blocks};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectedPair {
    pub target: usize,
    pub source: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairBlock {
    pub target_block: AtomBlock,
    pub pairs: Vec<DirectedPair>,
}

/// Partition symmetric pairs by their target atom block. Each pair is emitted
/// in both directions so a worker owns the force write for its target.
pub fn partition_directed_pairs(
    atom_count: usize,
    block_size: usize,
    pairs: &[(usize, usize)],
) -> Result<Vec<PairBlock>, String> {
    let blocks = make_atom_blocks(atom_count, block_size)?;
    let mut grouped: Vec<Vec<DirectedPair>> = vec![Vec::new(); blocks.len()];
    for &(a, b) in pairs {
        if a >= atom_count || b >= atom_count || a == b {
            return Err(format!("invalid pair ({a}, {b}) for {atom_count} atoms"));
        }
        let ab = atom_block_index(a, block_size)?;
        let bb = atom_block_index(b, block_size)?;
        grouped[ab].push(DirectedPair {
            target: a,
            source: b,
        });
        grouped[bb].push(DirectedPair {
            target: b,
            source: a,
        });
    }
    Ok(blocks
        .into_iter()
        .zip(grouped)
        .map(|(target_block, pairs)| PairBlock {
            target_block,
            pairs,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_pairs_are_owned_by_target_blocks() {
        let blocks = partition_directed_pairs(10, 4, &[(0, 7), (3, 8), (5, 6)]).unwrap();
        assert_eq!(blocks.len(), 3);
        for block in &blocks {
            assert!(
                block
                    .pairs
                    .iter()
                    .all(|p| block.target_block.contains(p.target))
            );
        }
        assert_eq!(
            blocks[0].pairs,
            vec![
                DirectedPair {
                    target: 0,
                    source: 7
                },
                DirectedPair {
                    target: 3,
                    source: 8
                },
            ]
        );
        assert_eq!(
            blocks[1].pairs,
            vec![
                DirectedPair {
                    target: 7,
                    source: 0
                },
                DirectedPair {
                    target: 5,
                    source: 6
                },
                DirectedPair {
                    target: 6,
                    source: 5
                },
            ]
        );
        assert_eq!(
            blocks[2].pairs,
            vec![DirectedPair {
                target: 8,
                source: 3
            }]
        );
    }

    #[test]
    fn rejects_invalid_pairs() {
        assert!(partition_directed_pairs(4, 2, &[(0, 4)]).is_err());
        assert!(partition_directed_pairs(4, 2, &[(1, 1)]).is_err());
    }
}
