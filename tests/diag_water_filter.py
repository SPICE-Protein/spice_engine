#!/usr/bin/env python
"""SE-side crystal-water filter (StructureInput path).

Regression for "Atom missing FF type" on 1R2I (214 crystal waters): the Tauri
GUI feeds the engine via `Structure.from_atoms` (StructureInput), whose
atoms_to_mmcif used to keep HOH waters as peptide residues → water O had no FF
type → hard crash. The engine now skips water residues there.
"""
import spice_engine as se
import numpy as np

PATH = "/Users/redelectricity/Documents/Projects/SPICE/SPICE_GUI/1R2I.cif"
WATER = {"HOH", "WAT", "SOL", "H2O", "DOD"}


def parse(path):
    atoms = []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t or t[0] not in ("ATOM", "HETATM") or len(t) < 14:
                continue
            atoms.append(
                {
                    "name": t[3],      # label_atom_id
                    "elem": t[2],      # type_symbol
                    "resname": t[5],   # label_comp_id
                    # 水分子行 label_seq_id 常为 "."；给占位 0（反正会被引擎过滤）
                    "seq": int(t[8]) if t[8] != "." else 0,
                    "x": float(t[10]), "y": float(t[11]), "z": float(t[12]),
                    "occ": float(t[13]),
                }
            )
    return atoms


atoms = parse(PATH)
waters = [a for a in atoms if a["resname"] in WATER]
print(f"parsed {len(atoms)} atoms, {len(waters)} waters", flush=True)

names = [a["name"] for a in atoms]
elems = [a["elem"] for a in atoms]
seqs = [a["seq"] for a in atoms]
resn = [a["resname"] for a in atoms]
coords = np.array([[a["x"], a["y"], a["z"]] for a in atoms], np.float32)
occ = np.array([a["occ"] for a in atoms], np.float32)

s = se.Structure.from_atoms(names, elems, seqs, resn, coords, occ)
print(f"from_atoms OK, residues: {s.residue_count()}", flush=True)

eng = se.Engine.build(
    s, 7.0, 310.0, 1.0, 0.0, relax_iters=30, tolerance=10.0, strict_incomplete=False
)
print("BUILD_OK (crystal waters filtered by SE)", flush=True)
