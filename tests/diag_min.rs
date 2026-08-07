//! Diagnostic (ignored): inspect post-minimization forces and the production-MD
//! blow-up mechanism, to guide the equilibration / minimization fixes.
//!
//! Two variants — run either:
//!   cargo test --release --test diag_min -- --ignored --nocapture diag_equil_off
//!   cargo test --release --test diag_min -- --ignored --nocapture diag_equil_on
//!
//! `diag_equil_off` shows the raw post-min state + production steps (the crash
//! mechanism). `diag_equil_on` additionally logs the equilibration ramp.
use bio_files::MmCif;
use dynamics::params::FfParamSet;
use dynamics::ComputationDevice;
use spice_engine::{BuildOptions, EnvParams, EquilConfig, build_system};
use std::path::Path;

/// (maxF, atom index, mass) of the worst protein atom.
fn max_f_prot(engine: &spice_engine::SpiceEngine) -> (f32, usize, f32) {
    let mut mf = 0.0f32;
    let mut idx = 0usize;
    let mut mass = 0.0f32;
    for (i, a) in engine.state.atoms.iter().enumerate() {
        let m = a.force.magnitude();
        if m > mf {
            mf = m;
            idx = i;
            mass = a.mass;
        }
    }
    (mf, idx, mass)
}

fn max_f_water(engine: &spice_engine::SpiceEngine) -> f32 {
    engine
        .state
        .water
        .iter()
        .flat_map(|w| {
            [
                w.m.force.magnitude(),
                w.h0.force.magnitude(),
                w.h1.force.magnitude(),
            ]
        })
        .fold(0.0f32, f32::max)
}

/// Print every S atom (mass≈32) with its neighbours within 2.5 Å — shows whether
/// disulfide pairs exist and whether any S has an anomalous bonded hydrogen.
fn print_sulfur_diag(engine: &spice_engine::SpiceEngine) {
    let s_atoms: Vec<usize> = engine
        .state
        .atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| (a.mass - 32.0).abs() < 1.0)
        .map(|(i, _)| i)
        .collect();
    println!("[sulfur] found {} S atoms", s_atoms.len());
    for &si in &s_atoms {
        let p = engine.state.atoms[si].posit;
        let mut nbrs: Vec<(usize, f32, f32)> = Vec::new();
        for (j, a) in engine.state.atoms.iter().enumerate() {
            if j == si {
                continue;
            }
            let d = (a.posit - p).magnitude();
            if d < 2.5 {
                nbrs.push((j, a.mass, d));
            }
        }
        nbrs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        let nbr_s = nbrs
            .iter()
            .find(|(_, m, _)| (*m - 32.0).abs() < 1.0)
            .map(|&(j, m, d)| format!("S-{}", j))
            .unwrap_or_else(|| "no-S-nbr".to_string());
        println!(
            "[sulfur] atom {si} ({p:.1}) force={:.1} nbrs={:?} -> {nbr_s}",
            engine.state.atoms[si].force.magnitude(),
            nbrs.iter()
                .map(|&(j, m, d)| format!("#{j}(m{m:.0},{d:.2})"))
                .collect::<Vec<_>>()
        );
    }
}

fn run(label: &str, equil: Option<EquilConfig>, pressure_bar: f32, n_prod_steps: usize) {
    let dev = ComputationDevice::Cpu;
    let param_set = FfParamSet::new_amber().expect("load amber params");
    let protein = MmCif::load(Path::new("data/test/2LYZ.cif")).expect("load 2LYZ mmCIF");
    let env = EnvParams {
        pressure_bar,
        ..Default::default()
    };
    let opts = BuildOptions {
        env,
        equil,
        ..Default::default()
    };

    let mut engine = build_system(&dev, &param_set, protein, &opts).expect("build system");

    print_sulfur_diag(&engine);

    // Count hard clashes in the built structure (protein atoms only, O(n²)).
    {
        let n = engine.state.atoms.len();
        let mut heavy = 0usize;
        let mut h_heavy = 0usize;
        let mut hh = 0usize;
        let mut worst: f32 = 1e9;
        let mut worst_pair: Vec<(usize, f32, f32, usize, f32, f32)> = Vec::new(); // (i,m_i,d,j,m_j,0)
        for i in 0..n {
            let ai = &engine.state.atoms[i];
            let ih = ai.mass < 2.0;
            for j in (i + 1)..n {
                let aj = &engine.state.atoms[j];
                let d = (ai.posit - aj.posit).magnitude();
                if d < worst {
                    worst = d;
                }
                let jh = aj.mass < 2.0;
                match (ih, jh) {
                    (false, false) if d < 1.3 => {
                        heavy += 1;
                        if d < 0.7 {
                            worst_pair.push((i, ai.mass, d, j, aj.mass, 0.0));
                        }
                    }
                    (true, false) | (false, true) if d < 0.9 => h_heavy += 1,
                    (true, true) if d < 0.8 => hh += 1,
                    _ => {}
                }
            }
        }
        println!(
            "[clash] heavy<1.3Å={heavy} H-heavy<0.9Å={h_heavy} H-H<0.8Å={hh} worst={worst:.3}Å n_prot={n}"
        );
        worst_pair.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        worst_pair.truncate(8);
        for (i, mi, d, j, mj, _) in worst_pair {
            let ri = engine
                .topology
                .residues
                .iter()
                .find(|r| r.atom_indices.contains(&i))
                .map(|r| format!("{}{}", r.one_letter, r.seq_id))
                .unwrap_or_else(|| "?".into());
            let rj = engine
                .topology
                .residues
                .iter()
                .find(|r| r.atom_indices.contains(&j))
                .map(|r| format!("{}{}", r.one_letter, r.seq_id))
                .unwrap_or_else(|| "?".into());
            println!(
                "[clashpair] {d:.3}Å atom{i}(m{mi:.0},{ri}) <-> atom{j}(m{mj:.0},{rj})"
            );
        }
    }

    // Diagnose atoms that repeatedly peak in the force traces.
    for &idx in &[343usize, 761, 1776, 1354] {
        if idx >= engine.state.atoms.len() {
            continue;
        }
        let p = engine.state.atoms[idx].posit;
        let res = engine
            .topology
            .residues
            .iter()
            .find(|r| r.atom_indices.contains(&idx))
            .map(|r| format!("{}{}", r.one_letter, r.seq_id))
            .unwrap_or_else(|| "?".into());
        let mut nbrs: Vec<(usize, f32, f32)> = Vec::new();
        for (j, a) in engine.state.atoms.iter().enumerate() {
            if j != idx {
                let d = (a.posit - p).magnitude();
                if d < 2.2 {
                    nbrs.push((j, a.mass, d));
                }
            }
        }
        nbrs.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        println!(
            "[atom{idx}] res={res} force={:.1} pos={p} nbrs={:?}",
            engine.state.atoms[idx].force.magnitude(),
            nbrs.iter()
                .map(|&(j, m, d)| format!("#{j}(m{m:.0},{d:.2})"))
                .collect::<Vec<_>>()
        );
    }

    let (mf, idx, mass) = max_f_prot(&engine);
    println!(
        "[{label}] post-min: maxF(protein)={mf:.1}(atom{idx},m{mass:.1}) maxF(water)={:.1} U={:.3e} n_water={} n_prot={}",
        max_f_water(&engine),
        engine.state.potential_energy,
        engine.state.water.len(),
        engine.state.atoms.len(),
    );

    let mut crashed_step: Option<usize> = None;
    for i in 0..n_prod_steps {
        let r = engine.step(None);
        let (mfp, pi, pm) = max_f_prot(&engine);
        let mfw = max_f_water(&engine);
        println!(
            "[{label}] step {i:2}: U={:.3e} maxF(prot)={mfp:10.1}(atom{pi},m{pm:.1}) maxF(water)={mfw:9.1}",
            r.u_t_kcal
        );
        if r.crashed {
            println!("[{label}] >>> CRASH at step {i}, U={:.3e}", r.u_t_kcal);
            crashed_step = Some(i);
            break;
        }
    }
    match crashed_step {
        Some(i) => println!("[{label}] RESULT: crashed at production step {i}"),
        None => println!("[{label}] RESULT: survived all {n_prod_steps} production steps"),
    }
}

#[test]
#[ignore]
fn diag_equil_off() {
    run("equil-off-bar-on", None, 1.0, 40);
}

#[test]
#[ignore]
fn diag_equil_off_nobar() {
    run("equil-off-nobar", None, 0.0, 40);
}

#[test]
#[ignore]
fn diag_equil_on() {
    run("equil-on-nobar", Some(EquilConfig::default()), 0.0, 40);
}

