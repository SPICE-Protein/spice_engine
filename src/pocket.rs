use std::collections::{VecDeque, HashSet};
use na_seq::Element;
use crate::engine::SpiceEngine;

/// Conversional scaling factor for partial charges in Amber force field (q_internal = q_e * scaler)
pub const CHARGE_UNIT_SCALER: f32 = 18.2223;

/// Grid status representations during the voxelization and pocket candidate selection phases.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum GridStatus {
    Empty = 0,
    Protein = 1,
    PocketCandidate = 2,
}

/// Geometric and topological representation of a detected ligand-binding cavity.
#[derive(Debug, Clone)]
pub struct NativePocket {
    pub id: usize,
    pub center: [f32; 3],              // 3D centroid of the pocket cavity
    pub volume: f32,                   // Volumetric size of the pocket (Å³)
    pub druggability: f32,             // Heuristic druggability score based on geometry and hydrophobicity
    pub surface_residues: Vec<usize>,  // 0-indexed indices of pocket-lining residues
    pub voxels: Vec<[f32; 3]>,         // Absolute 3D spatial coordinates of the constituent voxels (for MolStar rendering)
}

/// Rich physical, chemical, and pharmacophoric annotation of a binding pocket.
#[derive(Debug, Clone)]
pub struct AdvancedPocketFeatures {
    pub id: usize,
    pub center: [f32; 3],              // 3D centroid of the pocket cavity
    pub volume: f32,                   // Volumetric size of the pocket (Å³)
    pub druggability: f32,             // Integrated druggability score
    pub net_charge_at_ph: f32,         // Local net charge of the pocket wall at the current conditional pH (e)
    pub hydrophobic_ratio: f32,        // Fraction of hydrophobic contacts on the pocket surface (0.0 to 1.0)
    
    // 3D Pharmacophore Point Cloud
    pub hydrophobic_centroids: Vec<[f32; 3]>, 
    pub hbond_acceptors: Vec<([f32; 3], [f32; 3])>, // (Position [x, y, z], Direction vector [dx, dy, dz])
    pub hbond_donors: Vec<([f32; 3], [f32; 3])>,    // (Position [x, y, z], Direction vector [dx, dy, dz])
    
    pub surface_residues: Vec<usize>,  // 0-indexed indices of pocket-lining residues
}

/// Metric evaluating mutational perturbations on pocket volume and spatial displacement.
#[derive(Debug, Clone)]
pub struct PocketDelta {
    pub volume_delta: f32,             // Mutational volume shift (mutant_volume - reference_volume in Å³)
    pub jaccard_overlap: f32,          // Grid Jaccard similarity index measuring spatial translation (0.0 to 1.0)
}

/// Directional search operators for the LIGSITE ray-casting scan.
/// Comprises 3 orthogonal axes and 4 body diagonals of a cube.
const SCAN_DIRECTIONS: [[i32; 3]; 7] = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],               // Orthogonal axes (X, Y, Z)
    [1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1] // Cube diagonals
];

/// 100% Native Rust ligand-binding pocket calculation using grid-based voxelization and 7-directional ray casting.
///
/// # Arguments
/// * `coords` - Heavy atom coordinates (Å).
/// * `residue_indices` - Mapping from each heavy atom to its parent residue index.
/// * `is_hydrophobic` - Hydrophobicity flag for each atom.
/// * `grid_spacing` - Grid resolution (normally 1.0 Å).
pub fn calculate_pockets_native(
    coords: &[[f32; 3]],
    residue_indices: &[usize],
    is_hydrophobic: &[bool],
    grid_spacing: f32,
) -> Vec<NativePocket> {
    if coords.is_empty() || grid_spacing <= 0.0 {
        return vec![];
    }
    
    // 1. Compute protein bounding box and expand by a 4.0 Å solvent padding layer
    let mut min_p = [f32::MAX; 3];
    let mut max_p = [f32::MIN; 3];
    for p in coords {
        for i in 0..3 {
            if p[i] < min_p[i] { min_p[i] = p[i]; }
            if p[i] > max_p[i] { max_p[i] = p[i]; }
        }
    }
    
    let pad = 4.0;
    let origin = [min_p[0] - pad, min_p[1] - pad, min_p[2] - pad];
    let nx = ((max_p[0] - min_p[0] + 2.0 * pad) / grid_spacing).ceil() as usize;
    let ny = ((max_p[1] - min_p[1] + 2.0 * pad) / grid_spacing).ceil() as usize;
    let nz = ((max_p[2] - min_p[2] + 2.0 * pad) / grid_spacing).ceil() as usize;
    
    let grid_len = nx * ny * nz;
    let mut grid = vec![GridStatus::Empty; grid_len];
    
    let get_index = |x: usize, y: usize, z: usize| -> usize {
        x + y * nx + z * nx * ny
    };
    
    // 2. Fast voxelization: project protein atoms into the 3D grid
    // Combines the van der Waals radius and a water probe radius (~1.4 Å) to approximate solvent occlusion (~2.8 Å)
    let r_probe = 2.8;
    let r_grid = (r_probe / grid_spacing).ceil() as i32;
    
    for p in coords {
        let gx = ((p[0] - origin[0]) / grid_spacing) as i32;
        let gy = ((p[1] - origin[1]) / grid_spacing) as i32;
        let gz = ((p[2] - origin[2]) / grid_spacing) as i32;
        
        // Rasterize only within the local bounding sphere of the atom to achieve O(N_atoms * r_grid³) complexity
        for dx in -r_grid..=r_grid {
            for dy in -r_grid..=r_grid {
                for dz in -r_grid..=r_grid {
                    if dx*dx + dy*dy + dz*dz <= r_grid*r_grid {
                        let tx = gx + dx;
                        let ty = gy + dy;
                        let tz = gz + dz;
                        if tx >= 0 && tx < nx as i32 && ty >= 0 && ty < ny as i32 && tz >= 0 && tz < nz as i32 {
                            let idx = get_index(tx as usize, ty as usize, tz as usize);
                            grid[idx] = GridStatus::Protein;
                        }
                    }
                }
            }
        }
    }
    
    // 3. 7-Directional Ray Casting (LIGSITE Mechanism)
    // Identifies pocket candidates trapped in "cliffs/grooves" of the protein surface
    let max_scan_steps = (8.0 / grid_spacing) as i32; // Limit directional scans to 8.0 Å
    let mut candidates = Vec::new();
    for z in 1..(nz - 1) {
        for y in 1..(ny - 1) {
            for x in 1..(nx - 1) {
                let center_idx = get_index(x, y, z);
                if grid[center_idx] != GridStatus::Empty { continue; }
                
                let mut sandwich_count = 0;
                for dir in &SCAN_DIRECTIONS {
                    let mut hit_positive = false;
                    let mut hit_negative = false;
                    
                    // Positive direction ray
                    for step in 1..=max_scan_steps {
                        let tx = x as i32 + dir[0] * step;
                        let ty = y as i32 + dir[1] * step;
                        let tz = z as i32 + dir[2] * step;
                        if tx < 0 || tx >= nx as i32 || ty < 0 || ty >= ny as i32 || tz < 0 || tz >= nz as i32 { break; }
                        if grid[get_index(tx as usize, ty as usize, tz as usize)] == GridStatus::Protein {
                            hit_positive = true;
                            break;
                        }
                    }
                    
                    // Negative direction ray
                    for step in 1..=max_scan_steps {
                        let tx = x as i32 - dir[0] * step;
                        let ty = y as i32 - dir[1] * step;
                        let tz = z as i32 - dir[2] * step;
                        if tx < 0 || tx >= nx as i32 || ty < 0 || ty >= ny as i32 || tz < 0 || tz >= nz as i32 { break; }
                        if grid[get_index(tx as usize, ty as usize, tz as usize)] == GridStatus::Protein {
                            hit_negative = true;
                            break;
                        }
                    }
                    
                    if hit_positive && hit_negative {
                        sandwich_count += 1;
                    }
                }
                
                // If a grid voxel is sandwiched in 4 or more out of 7 directions, mark it as pocket candidate
                if sandwich_count >= 4 {
                    grid[center_idx] = GridStatus::PocketCandidate;
                    candidates.push((x, y, z));
                }
            }
        }
    }
    
    // 4. 26-Neighborhood Breadth-First Search (BFS) Clustering
    // Clusters adjacent candidate voxels into distinct topological cavities
    let mut visited = vec![false; grid_len];
    let mut pockets = Vec::new();
    let mut pocket_id = 0;
    
    for &(cx, cy, cz) in &candidates {
        let start_idx = get_index(cx, cy, cz);
        if visited[start_idx] { continue; }
        
        let mut cluster = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((cx, cy, cz));
        visited[start_idx] = true;
        
        while let Some((ux, uy, uz)) = queue.pop_front() {
            cluster.push((ux, uy, uz));
            
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 { continue; }
                        let tx = ux as i32 + dx;
                        let ty = uy as i32 + dy;
                        let tz = uz as i32 + dz;
                        if tx >= 0 && tx < nx as i32 && ty >= 0 && ty < ny as i32 && tz >= 0 && tz < nz as i32 {
                            let n_idx = get_index(tx as usize, ty as usize, tz as usize);
                            if grid[n_idx] == GridStatus::PocketCandidate && !visited[n_idx] {
                                visited[n_idx] = true;
                                queue.push_back((tx as usize, ty as usize, tz as usize));
                            }
                        }
                    }
                }
            }
        }
        
        // 5. Pocket filtering: discard trivial cavities (antagonists typically require volume > 250 Å³)
        let voxel_volume = grid_spacing.powi(3);
        let volume = cluster.len() as f32 * voxel_volume;
        if volume >= 200.0 {
            let mut sum_coord = [0.0; 3];
            let mut voxels = Vec::with_capacity(cluster.len());
            for &(ux, uy, uz) in &cluster {
                let vx = origin[0] + ux as f32 * grid_spacing;
                let vy = origin[1] + uy as f32 * grid_spacing;
                let vz = origin[2] + uz as f32 * grid_spacing;
                sum_coord[0] += vx;
                sum_coord[1] += vy;
                sum_coord[2] += vz;
                voxels.push([vx, vy, vz]);
            }
            let len_f = cluster.len() as f32;
            let center = [sum_coord[0] / len_f, sum_coord[1] / len_f, sum_coord[2] / len_f];
            
            // 6. Identify pocket-lining surface residues and local hydrophobicity ratio
            let mut pocket_residues = Vec::new();
            let mut hydrophobic_count = 0;
            let mut total_contact_atoms = 0;
            
            for (idx, p) in coords.iter().enumerate() {
                let dx = p[0] - center[0];
                let dy = p[1] - center[1];
                let dz = p[2] - center[2];
                let dist_sq = dx*dx + dy*dy + dz*dz;
                
                // Atoms within 6.5 Å of the centroid are treated as pocket surface lining
                if dist_sq <= 6.5 * 6.5 {
                    total_contact_atoms += 1;
                    if is_hydrophobic[idx] {
                        hydrophobic_count += 1;
                    }
                    let res_idx = residue_indices[idx];
                    if !pocket_residues.contains(&res_idx) {
                        pocket_residues.push(res_idx);
                    }
                }
            }
            
            let hydrophobic_ratio = if total_contact_atoms > 0 {
                hydrophobic_count as f32 / total_contact_atoms as f32
            } else {
                0.0
            };
            
            // Antagonists rely heavily on entropic hydrophobic shielding; druggability scales with volume and hydrophobicity
            let druggability = (volume / 1200.0).min(1.0) * 0.5 + hydrophobic_ratio * 0.5;
            
            pockets.push(NativePocket {
                id: pocket_id,
                center,
                volume,
                druggability,
                surface_residues: pocket_residues,
                voxels,
            });
            pocket_id += 1;
        }
    }
    
    // Sort pockets in descending order of volume (largest cavity is normally the primary binding site)
    pockets.sort_by(|a, b| b.volume.partial_cmp(&a.volume).unwrap_or(std::cmp::Ordering::Equal));
    pockets
}

/// Helper function to check if an element in a given amino acid residue is hydrophobic.
pub fn is_atom_hydrophobic(element: Element, one_letter: char) -> bool {
    match element {
        Element::Carbon | Element::Sulfur => {
            matches!(one_letter, 'A' | 'V' | 'L' | 'I' | 'M' | 'F' | 'W' | 'P' | 'Y' | 'C')
        }
        _ => false,
    }
}

/// High-level API to calculate protein pockets directly from a `SpiceEngine`.
pub fn calculate_engine_pockets(engine: &SpiceEngine, grid_spacing: f32) -> Vec<NativePocket> {
    let heavy = &engine.topology.heavy_indices;
    if heavy.is_empty() {
        return vec![];
    }

    let mut atom_to_res = vec![usize::MAX; engine.state.atoms.len()];
    for (res_idx, residue) in engine.topology.residues.iter().enumerate() {
        for &atom_idx in &residue.atom_indices {
            if atom_idx < atom_to_res.len() {
                atom_to_res[atom_idx] = res_idx;
            }
        }
    }

    let mut coords = Vec::with_capacity(heavy.len());
    let mut residue_indices = Vec::with_capacity(heavy.len());
    let mut is_hydrophobic = Vec::with_capacity(heavy.len());

    for &ai in heavy {
        let p = engine.state.atoms[ai].posit;
        coords.push([p.x, p.y, p.z]);
        
        let res_idx = atom_to_res[ai];
        residue_indices.push(if res_idx == usize::MAX { 0 } else { res_idx });
        
        let one_letter = if res_idx != usize::MAX {
            engine.topology.residues[res_idx].one_letter
        } else {
            'X'
        };
        let elem = engine.state.atoms[ai].element;
        is_hydrophobic.push(is_atom_hydrophobic(elem, one_letter));
    }

    calculate_pockets_native(&coords, &residue_indices, &is_hydrophobic, grid_spacing)
}

/// Calculate advanced dynamic, electrostatic, and pharmacophoric features of a pocket.
pub fn calculate_advanced_features(
    engine: &SpiceEngine,
    pocket: &NativePocket,
) -> AdvancedPocketFeatures {
    let mut atom_to_res = vec![usize::MAX; engine.state.atoms.len()];
    for (res_idx, residue) in engine.topology.residues.iter().enumerate() {
        for &atom_idx in &residue.atom_indices {
            if atom_idx < atom_to_res.len() {
                atom_to_res[atom_idx] = res_idx;
            }
        }
    }

    // 1. Compute local pocket-lining net charge in elementary charge units (e)
    let mut net_charge = 0.0f32;
    for atom in &engine.state.atoms {
        let dx = atom.posit.x - pocket.center[0];
        let dy = atom.posit.y - pocket.center[1];
        let dz = atom.posit.z - pocket.center[2];
        if dx*dx + dy*dy + dz*dz <= 6.5 * 6.5 {
            net_charge += atom.partial_charge / CHARGE_UNIT_SCALER;
        }
    }

    // 2. Map 3D pharmacophore points (Hydrophobic centroids, Hydrogen-bond donors & acceptors)
    let mut hydrophobic_centroids = Vec::new();
    let mut hbond_acceptors = Vec::new();
    let mut hbond_donors = Vec::new();

    for &res_idx in &pocket.surface_residues {
        if res_idx >= engine.topology.residues.len() { continue; }
        let residue = &engine.topology.residues[res_idx];
        
        // A. Hydrophobic Centroid computation for non-polar residues
        if matches!(residue.one_letter, 'A' | 'V' | 'L' | 'I' | 'M' | 'F' | 'W' | 'P' | 'Y' | 'C') {
            let mut sum_pos = [0.0f32; 3];
            let mut count = 0;
            for &ai in &residue.atom_indices {
                let atom = &engine.state.atoms[ai];
                if matches!(atom.element, Element::Carbon | Element::Sulfur) {
                    sum_pos[0] += atom.posit.x;
                    sum_pos[1] += atom.posit.y;
                    sum_pos[2] += atom.posit.z;
                    count += 1;
                }
            }
            if count > 0 {
                hydrophobic_centroids.push([
                    sum_pos[0] / count as f32,
                    sum_pos[1] / count as f32,
                    sum_pos[2] / count as f32,
                ]);
            }
        }

        // B. H-bond Acceptor and Donor vector detection using polar covalent geometries
        for &ai in &residue.atom_indices {
            let atom = &engine.state.atoms[ai];
            
            // Oxygens act as H-bond acceptors
            if atom.element == Element::Oxygen {
                let mut nearest_neighbor: Option<usize> = None;
                let mut min_d2 = 1.7 * 1.7; // Standard covalent bond length cutoff
                for &oi in &residue.atom_indices {
                    if oi == ai { continue; }
                    let neighbor = &engine.state.atoms[oi];
                    if neighbor.element != Element::Hydrogen {
                        let dx = neighbor.posit.x - atom.posit.x;
                        let dy = neighbor.posit.y - atom.posit.y;
                        let dz = neighbor.posit.z - atom.posit.z;
                        let d2 = dx*dx + dy*dy + dz*dz;
                        if d2 < min_d2 {
                            min_d2 = d2;
                            nearest_neighbor = Some(oi);
                        }
                    }
                }
                
                if let Some(ni) = nearest_neighbor {
                    let neighbor = &engine.state.atoms[ni];
                    let dx = atom.posit.x - neighbor.posit.x;
                    let dy = atom.posit.y - neighbor.posit.y;
                    let dz = atom.posit.z - neighbor.posit.z;
                    let len = (dx*dx + dy*dy + dz*dz).sqrt();
                    if len > 0.0 {
                        hbond_acceptors.push((
                            [atom.posit.x, atom.posit.y, atom.posit.z],
                            [dx / len, dy / len, dz / len]
                        ));
                    }
                }
            }

            // Nitrogens and Oxygens with bonded hydrogens act as H-bond donors
            if matches!(atom.element, Element::Nitrogen | Element::Oxygen) {
                for &hi in &residue.atom_indices {
                    let h_atom = &engine.state.atoms[hi];
                    if h_atom.element == Element::Hydrogen {
                        let dx = h_atom.posit.x - atom.posit.x;
                        let dy = h_atom.posit.y - atom.posit.y;
                        let dz = h_atom.posit.z - atom.posit.z;
                        let dist_sq = dx*dx + dy*dy + dz*dz;
                        let limit = if atom.element == Element::Nitrogen { 1.25 * 1.25 } else { 1.15 * 1.15 };
                        if dist_sq <= limit {
                            let len = dist_sq.sqrt();
                            if len > 0.0 {
                                hbond_donors.push((
                                    [atom.posit.x, atom.posit.y, atom.posit.z],
                                    [dx / len, dy / len, dz / len]
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    let total_contact_atoms = pocket.surface_residues.iter()
        .map(|&ri| if ri < engine.topology.residues.len() { engine.topology.residues[ri].atom_indices.len() } else { 0 })
        .sum::<usize>();

    let mut hydrophobic_contact_atoms = 0;
    for &res_idx in &pocket.surface_residues {
        if res_idx >= engine.topology.residues.len() { continue; }
        let res = &engine.topology.residues[res_idx];
        if matches!(res.one_letter, 'A' | 'V' | 'L' | 'I' | 'M' | 'F' | 'W' | 'P' | 'Y' | 'C') {
            hydrophobic_contact_atoms += res.atom_indices.iter()
                .filter(|&&ai| matches!(engine.state.atoms[ai].element, Element::Carbon | Element::Sulfur))
                .count();
        }
    }

    let hydrophobic_ratio = if total_contact_atoms > 0 {
        hydrophobic_contact_atoms as f32 / total_contact_atoms as f32
    } else {
        0.0
    };

    AdvancedPocketFeatures {
        id: pocket.id,
        center: pocket.center,
        volume: pocket.volume,
        druggability: pocket.druggability,
        net_charge_at_ph: net_charge,
        hydrophobic_ratio,
        hydrophobic_centroids,
        hbond_acceptors,
        hbond_donors,
        surface_residues: pocket.surface_residues.clone(),
    }
}

/// Computes the mutational pocket spatial perturbation (Delta volume and Jaccard coordinate intersection).
pub fn calculate_pocket_delta(
    ref_pocket: &NativePocket,
    mut_pocket: &NativePocket,
    grid_spacing: f32,
) -> PocketDelta {
    if grid_spacing <= 0.0 {
        return PocketDelta { volume_delta: 0.0, jaccard_overlap: 0.0 };
    }

    let mut ref_set = HashSet::new();
    for v in &ref_pocket.voxels {
        let key = (
            (v[0] / grid_spacing).round() as i32,
            (v[1] / grid_spacing).round() as i32,
            (v[2] / grid_spacing).round() as i32,
        );
        ref_set.insert(key);
    }

    let mut intersection = 0usize;
    let mut union_set = ref_set.clone();

    for v in &mut_pocket.voxels {
        let key = (
            (v[0] / grid_spacing).round() as i32,
            (v[1] / grid_spacing).round() as i32,
            (v[2] / grid_spacing).round() as i32,
        );
        if ref_set.contains(&key) {
            intersection += 1;
        }
        union_set.insert(key);
    }

    let jaccard_overlap = if union_set.is_empty() {
        0.0
    } else {
        intersection as f32 / union_set.len() as f32
    };

    PocketDelta {
        volume_delta: mut_pocket.volume - ref_pocket.volume,
        jaccard_overlap,
    }
}

/// Analyze pocket volume fluctuations (MAD) and open probability dynamically along an online trajectory.
pub fn analyze_pocket_trajectory(
    engine: &mut SpiceEngine,
    samples: usize,
    step_size: usize,
    grid_spacing: f32,
    threshold_volume: f32,
) -> (f32, f32) {
    let mut volumes = Vec::new();
    for _ in 0..samples {
        for _ in 0..step_size {
            engine.step(None);
        }
        let pockets = calculate_engine_pockets(engine, grid_spacing);
        let vol = if pockets.is_empty() { 0.0 } else { pockets[0].volume };
        volumes.push(vol);
    }
    
    if volumes.is_empty() {
        return (0.0, 0.0);
    }
    
    let mut sorted_v = volumes.clone();
    sorted_v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted_v[sorted_v.len() / 2];
    
    let mut diffs: Vec<f32> = volumes.iter().map(|&v| (v - median).abs()).collect();
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad_diff = diffs[diffs.len() / 2];
    let mad = mad_diff * 1.4826;
    
    let open_count = volumes.iter().filter(|&&v| v > threshold_volume).count();
    let open_prob = open_count as f32 / volumes.len() as f32;
    
    (mad, open_prob)
}
