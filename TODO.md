# SPICE Engine TODO

## v1.2 性能大增长

### 基线与测量
- [x] 记录一次 `cargo test --release --test md_smoke -- --nocapture` 基线。
- [x] 增加独立 release benchmark：排除首次编译、建系、溶剂初始化和 metrics，只测连续 MD step。
- [x] 记录 100 步 wall mean/P50/P95/P99 与 dynamics 分项；已增加 ordinary/PME/rebuild 分类均值。
- [x] benchmark 输出 build time、phase timing、分类计数、virial 分项和 pair 覆盖率。

### CPU SIMD
- [x] 建立 SE SIMD backend 探测接口。
- [x] x86_64 接入 AVX2/AVX-512 runtime capability detection。
- [x] arm64 接入 NEON/ASIMD baseline backend detection。
- [x] 将基础 LJ/Coulomb 公式实现为 SE SIMD-friendly 8-lane kernel（实验性，尚未进入 MdState 热路径）。
- [x] x86_64 普通 std-std pair 接入当前 `dynamics` nonbonded 热路径（本地 fork 临时 x8 dispatch；最终仍需迁移到 SE）。
- [ ] 实现 x86_64 AVX-512 x16 nonbonded kernel；当前仅有 SE 侧 kernel 骨架，target-owned 延迟 source 归约实验因 rebuild/轨迹回归已回退。
- [x] arm64 NEON 基础 pair microbenchmark；生产路径使用保守 SIMD LJ + 精确 scalar Coulomb。
- [x] 增加 SIMD/scalar 基础 pair force/energy 对照测试；microbenchmark checksum 对照误差已验证，完整 virial/生产路径对照仍待补。
- [x] 处理 SIMD tail lanes 和非整批 pair（在邻居表构建阶段转入 scalar cache）。

### 数据与邻居表
- [x] SE 侧 `CsrNeighborList` 已完成；dynamics 生产 std/water neighbor 遍历已接入 CSR 镜像，旧 Vec<Vec> 仍保留作兼容回退。
- [x] SIMD-ready pair 已预展开为 numeric metadata；索引已压缩为 u32，scalar 特殊 pair 仍保留 `NonBondedPair`。
- [ ] 预计算 sigma、epsilon、charge product 和 scaling flags。
- [x] SE 侧新增 `AtomBlock`、`BlockForceAccumulator` 和 directed `PairBlock` 分区；dynamics 生产 target-owned force accumulation 实验已回退，避免 rebuild/轨迹回归。
- [ ] benchmark skin/cutoff/rebuild trade-off。

### PME/长程
- [x] benchmark 已按实际标志区分 PME 与 PME+rebuild，并按类别输出 wall/virial 均值，且 warm-up 后固定速度温度。
- [ ] benchmark mesh spacing、alpha 和 SPME ratio。
- [ ] 检查 PME cache 对 energy/virial/pressure 的影响。

### 上层开销
- [ ] 增加 fast step API，允许低频计算 metrics。
- [ ] 减少 Cα 坐标和 FFI 对象分配。
- [ ] 分离 metrics timing 和 MD timing。

## 力场与结构

- [ ] 完成 SE 内部 Amber 参数解析和索引化。
- [ ] 完成 CHARMM36m 参数适配器。
- [ ] 完成 Martini 3 mapping/topology，不把 bead 直接当全原子使用。
- [ ] 接通 Protein/RNA/Ligand/Lipid 的内容路由。
- [ ] 保持蛋白质现有构象重建路径不变。
- [ ] 为 S4RNA 接入“已有/模型生成构象 → Engine RNA topology”路径。

## 明确不做

- [x] 暂不加入 GPU/CUDA 后端。
- [x] 不在 SPICE Engine 实现 RNA folding prediction。
- [x] 不把 ML force field 或 ML 参数模型放入 Engine。
