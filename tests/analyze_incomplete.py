#!/usr/bin/env python
"""Analyze incomplete (backbone-only) residues in 1R2I.cif."""
import sys
from collections import defaultdict

path = "data/test/1R2I.cif"
BB = {"N", "CA", "C", "O"}

# Parse atom_site: group by label_seq_id ($9). Columns: 1 group, 2 id, 3 elem,
# 4 label_atom_id, 5 alt, 6 comp, 7 chain, 8 entity, 9 label_seq_id, 10 x, 11 y, 12 z
res_atoms = defaultdict(list)  # label_seq_id -> list of atom names
res_chain = {}
with open(path) as f:
    in_loop = False
    headers = []
    for line in f:
        t = line.strip()
        if t == "loop_":
            in_loop = True
            headers = []
            continue
        if in_loop and t.startswith("_"):
            headers.append(t)
            continue
        if in_loop and headers and headers[0].startswith("_atom_site."):
            if t == "#" or t.startswith("_") or t == "loop_":
                in_loop = False
                continue
            flds = t.split()
            if len(flds) < 9 or flds[0] not in ("ATOM", "HETATM"):
                continue
            # find column indices
            def col(tag):
                return headers.index(tag) if tag in headers else -1
            c_res = col("_atom_site.label_seq_id")
            c_chain = col("_atom_site.label_asym_id")
            c_name = col("_atom_site.label_atom_id")
            c_group = col("_atom_site.group_PDB")
            c_el = col("_atom_site.type_symbol")
            if c_res < 0 or c_chain < 0 or c_name < 0:
                continue
            if flds[c_group] != "ATOM":
                continue
            res = flds[c_res]
            res_atoms[res].append(flds[c_name])
            res_chain.setdefault(res, flds[c_chain])

backbone_only = []
for res, names in sorted(res_atoms.items(), key=lambda kv: int(kv[0])):
    heavy = {n for n in names if not n.startswith("H")}
    if heavy and heavy <= BB:
        backbone_only.append((res, res_chain[res], sorted(heavy)))

print(f"total residues: {len(res_atoms)}")
print(f"backbone-only residues: {len(backbone_only)}")
for r in backbone_only:
    print(f"  res {r[0]} chain {r[1]} atoms={r[2]}")
