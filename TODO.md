# SPICE Engine TODO

## 当前架构决策（参考 LAMMPS）

- Engine 负责物理计算；SPICE Model/S4RNA 负责折叠与构象策略。
- 普通非键合相互作用采用 half-neighbor 语义：每条无向 pair 只计算一次，force 双向更新，energy/virial 单计数。
- Pair style 按物理内容分组：standard LJ+Coulomb、1-4、water-standard、water-water、alchemical；不在 SIMD 热循环内反复做枚举分支。
- 并行采用 worker-local force/energy/virial，再按固定顺序归约；禁止直接共享写 atom force，也不使用未经验证的 atomic force。
- CSR neighbor list 是目标布局；当前 dynamics 保留旧 `Vec<Vec>` 兼容层，逐步切换。
- `ForceField`/`ForceFieldSystem` 使用 SE-owned 类型；蛋白质现有构象重建路径保持不动。

## 当前性能基线（arm64 本机）

- 100-step detailed benchmark：普通 PME step 约 25–30 ms，PME+neighbor-rebuild 约 75–105 ms。
- 主要直接成本：short-range nonbonded 约 16–23 ms，ambient/water 约 20–30 ms，neighbor rebuild 约 4–10 ms。
- 8-lane NEON/portable pair microbenchmark 约 4–5x scalar；整步性能不能用 microbenchmark 代替。
- 任何优化必须同时检查：energy、virial、neighbor-rebuild 次数、crash/stability、P50/P95/P99。


## v1.2 性能大增长

### 基线与测量
- [x] 记录一次 `cargo test --release --test md_smoke -- --nocapture` 基线。
- [x] 增加独立 release benchmark：排除首次编译、建系、溶剂初始化和 metrics，只测连续 MD step。
- [x] 记录 100 步 wall mean/P50/P95/P99 与 dynamics 分项；已增加 ordinary/PME/rebuild 分类均值。v1.2 checkpoint 后基线：mean 37.401 ms，P50 28.179 ms，P95 109.475 ms。
- [x] benchmark 输出 build time、phase timing、分类计数、virial 分项和 pair 覆盖率；checkpoint 后 pair：SIMD 330544、scalar 873837。

### CPU SIMD
- [x] 建立 SE SIMD backend 探测接口。
- [x] x86_64 接入 AVX2/AVX-512 runtime capability detection。
- [x] arm64 接入 NEON/ASIMD baseline backend detection。
- [x] 将基础 LJ/Coulomb 公式实现为 SE SIMD-friendly 8-lane kernel（实验性，尚未进入 MdState 热路径）。
- [x] x86_64 普通 std-std pair 接入当前 `dynamics` nonbonded 热路径（本地 fork 临时 x8 dispatch；最终仍需迁移到 SE）。
- [x] 实现 x86_64 AVX-512 x16 nonbonded kernel；已加入安全 runtime dispatch、逐 lane scalar 对照和 finite 检查；完整 MdState 生产热路径仍未强行接入。
- [x] arm64 NEON 基础 pair microbenchmark；生产路径使用保守 SIMD LJ + 精确 scalar Coulomb。
- [x] 增加 SIMD/scalar 基础 pair force/energy 对照测试；microbenchmark checksum 对照误差已验证，完整 virial/生产路径对照仍待补。
- [x] 处理 SIMD tail lanes 和非整批 pair（在邻居表构建阶段转入 scalar cache）。

### 数据与邻居表
- [x] SE 侧 `CsrNeighborList` 已完成；dynamics 生产 std/water neighbor 遍历已接入 CSR 镜像，旧 Vec<Vec> 仍保留作兼容回退。
- [x] SIMD-ready pair 已预展开为 numeric metadata；索引已压缩为 u32，scalar 特殊 pair 仍保留 `NonBondedPair`。
- [x] SIMD-ready pair 预计算 sigma、epsilon、charge product；scaling flags 已在筛选阶段固定，scalar 特殊 pair 仍保留原 flags。
- [x] SE 侧新增 `AtomBlock`、`BlockForceAccumulator` 和 directed `PairBlock` 分区；dynamics 生产 target-owned force accumulation 实验已回退，避免 rebuild/轨迹回归。
- [x] 增加 skin/cutoff/rebuild trade-off benchmark；`BuildOptions` 暴露 `neighbor_skin`，benchmark 可扫描 skin 参数并记录 rebuild/phase/wall；dynamics 已复用 neighbor/CSR 缓冲。

### PME/长程
- [x] benchmark 已按实际标志区分 PME 与 PME+rebuild，并按类别输出 wall/virial 均值，且 warm-up 后固定速度温度。
- [x] 增加 PME mesh spacing/alpha 参数 sweep benchmark；SPME ratio 以当前实现的参数化 sweep 记录，完整独立 ratio 扫描仍需后续补充。
- [x] 检查 PME cache 对 energy/virial/pressure 的影响；新增 `benchmark_pme_cache_regression`，验证当前 refresh-every-step 策略的 PME、energy、long-range virial 有限性与无意外 neighbor rebuild。cache-off 对照不适用当前 dynamics API。

### 上层开销
- [x] 增加 `step_fast` API，支持 `metrics_every=0` 跳过 metrics 或按固定步长低频计算。
- [x] 减少 Cα 坐标和 FFI 对象分配；`coords_scratch`/`flat_coords_scratch` 复用，新增 `coords_ca_flat()` 避免 nested `Vec<Vec<f32>>` 中间对象。NumPy 所有权复制仍不可避免。
- [x] 分离 metrics timing 和 MD timing；`computation_time()` 增加 `last_md_us` 与 `last_metrics_us`。

## 力场与结构

- [x] 完成 SE-owned `ForceFieldSystem` 的基础 half-neighbor LJ+Coulomb 求值、Newton 双向力更新和 energy/virial 单计数；完整 Amber 参数解析和索引化仍待完成。
- [x] 完成 SE 内部 Amber 参数解析和索引化的基础索引：MASS/BOND/ANGL/DIHE/NONBON 记录与 canonical lookup；完整 frcmod 覆盖和 bonded 求值仍待完成。
- [x] 完成 CHARMM36m 基础 adapter（参数资产、内容选择和 SE-owned 求值边界）；完整 CHARMM 拓扑/参数语义解析仍待完成。
- [x] 完成 Martini 3 mapping/topology 的 SE-owned 基础解析与验证：atom→bead assignment、bead 记录、bond/constraint topology；实际 coarse-grained 坐标构建仍待完成。
- [x] 接通 Protein/RNA/Ligand/Lipid 的 force-field content selection/compatibility 路由；RNA 实际 topology build 仍由 S4RNA 项目项覆盖。
- [x] 保持蛋白质现有构象重建路径不变；新增 `tests/topology_regression.rs`，覆盖 prepared MmCif、alternate-conformer 去重、氢处理和稳定 Cα/topology 索引。
- [ ] 为 S4RNA 接入“已有/模型生成构象 → Engine RNA topology”路径。

## 明确不做

- [x] 暂不加入 GPU/CUDA 后端。
- [x] 不在 SPICE Engine 实现 RNA folding prediction。
- [x] 不把 ML force field 或 ML 参数模型放入 Engine。
