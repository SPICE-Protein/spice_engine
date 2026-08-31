//! Compact CSR neighbor-list representation for the SE CPU path.
//!
//! The current dynamics compatibility path still uses `Vec<Vec<usize>>`; this
//! type is the migration target and keeps neighbor indices contiguous for SIMD
//! batching and block ownership.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CsrNeighborList {
    /// Row offsets; row `i` is `indices[offsets[i]..offsets[i + 1]]`.
    pub offsets: Vec<u32>,
    /// Contiguous neighbor indices.
    pub indices: Vec<u32>,
}

impl CsrNeighborList {
    pub fn from_rows(rows: &[Vec<usize>]) -> Result<Self, String> {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        let mut indices = Vec::new();
        offsets.push(0);
        for (row, neighbors) in rows.iter().enumerate() {
            for &neighbor in neighbors {
                let value = u32::try_from(neighbor)
                    .map_err(|_| format!("neighbor index {neighbor} in row {row} exceeds u32"))?;
                indices.push(value);
            }
            let offset = u32::try_from(indices.len())
                .map_err(|_| "CSR neighbor list exceeds u32 address space".to_string())?;
            offsets.push(offset);
        }
        Ok(Self { offsets, indices })
    }

    pub fn row_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn row(&self, row: usize) -> Option<&[u32]> {
        let start = *self.offsets.get(row)? as usize;
        let end = *self.offsets.get(row + 1)? as usize;
        self.indices.get(start..end)
    }

    pub fn iter_rows(&self) -> impl Iterator<Item = &[u32]> {
        (0..self.row_count()).map(|row| {
            let start = self.offsets[row] as usize;
            let end = self.offsets[row + 1] as usize;
            &self.indices[start..end]
        })
    }

    pub fn total_neighbors(&self) -> usize {
        self.indices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_rows_and_order() {
        let rows = vec![vec![3, 1], vec![], vec![0, 2, 4]];
        let csr = CsrNeighborList::from_rows(&rows).unwrap();
        assert_eq!(csr.offsets, vec![0, 2, 2, 5]);
        assert_eq!(csr.indices, vec![3, 1, 0, 2, 4]);
        assert_eq!(csr.row_count(), 3);
        assert_eq!(csr.row(0), Some(&[3, 1][..]));
        assert_eq!(csr.row(1), Some(&[][..]));
        assert_eq!(
            csr.iter_rows().collect::<Vec<_>>(),
            vec![&[3, 1][..], &[][..], &[0, 2, 4][..]]
        );
    }

    #[test]
    fn rejects_indices_that_do_not_fit() {
        let rows = vec![vec![usize::MAX]];
        assert!(CsrNeighborList::from_rows(&rows).is_err());
    }
}
