# P7 / T02：Sampling Params Gate 与 M10 语义 — 实现 Spec

```text
spec_version:   p7-t02-spec.v1
task:           T02（Sampling Params Gate 与 M10 数值合同）
contract_ref:   p7-task-contract.v3
owner:          <待回填>
reviewer:       T09
stage:          WS1 主实现；WS2 延续（逐 rank canonical record 校验）
code_baseline:  origin/main @ 4406afc（T01 冻结基线公布后需 rebase，见 §9）
status:         draft（消费的 T01 冻结物未到位处已标注 [待 T01]）
```

> 本文件是 T02 的内部实现 Spec，不是对外合同。发生冲突时，以 `p7-task-contract.v3`、配套 schema、
> fixture manifest 和测试命令为准。本 Spec 只把合同 **T02 章节**（`p7-task-selection.md#L197-L223`）
> 和合同 **§二 共享 M10 输入/输出合同**（`#L43-L80`）落成可实现的字段与决策规则。

---

## 1. 目标与范围

**一句话目标：** 确保训练侧和 rollout 侧都使用同一套 **raw model logprob** 语义；任意非 identity
sampling 设置、实际语义漂移、LM Head LoRA 或未声明 backend 都必须在**计算/比较之前**明确失败。

**范围内：**

- canonical sampling record 的规范化与四层校验（requested / normalized / materialized / runtime actual）。
- M10 数值合同的"调用现有 `linear_logp` + 校验语义"部分（不是重新实现 GEMM / log-softmax / `dlogits`）。
- 参数 provenance 字段与正/负向测试。
- WS2 延续：逐 rank canonical sampling record 的完整性与一致性校验（本 Spec 定义字段，实现留 WS2）。

**范围外（合同明确不属于 P7/T02）：**

- 不新写 GEMM kernel、TP LM Head、log-softmax 或 `dlogits` backward（`#L30`、`#L214`）。
- 不实现 sampling transform 的重放（一致性模式要求 identity，主路径不重放，`#L33`）。
- 不实现 expert LoRA（只记录并校验其 identity；`#L32`）。
- 不物化完整 `[token, vocab]` logits（`#L30`）。

---

## 2. 依赖与基线

| 依赖 | 状态 | 说明 |
| --- | --- | --- |
| T01 冻结的 M10 输入/输出 schema | [待 T01] | 合同 `#L43-L80` 已给字段清单，本 Spec 按其具体化；正式 dtype/shape/layout 以 T01 manifest 为准 |
| T01 boundary-contract manifest（M10 行） | [待 T01] | `comparison_mode`、`tolerance_contract_version`、fixture checksum |
| T01 冻结的 canonical hash version | [待 T01] | 用于 `boundary_hash_index` |
| P1 共享 fixed-K/batch-invariant GEMM arithmetic spec + bit-equivalence harness | [待 T01] | `#L173`；用于验证现有 `linear_logp` 的 LM Head 路径 |
| T01 provider stub + synthetic fixture + CPU oracle | [待 T01] | `#L168`；未到位前用本 Spec §7 的自建 stub 顶替 |
| T05/T06 真 provider 的 actual readback | WS2 用 | 只影响 WS2 的 actual 校验，不阻塞 WS1（`#L22`） |

**基线结论：** 只需 T01 start kit 即可开工（`#L601`）。T05/T06 的真 readback 在 WS1 不阻塞。

### 2.1 H20/SM90 预验证证据（非正式 gate 证据）

本机为 `NVIDIA H20`（compute capability **9,0** = sm_90），CUDA toolkit 12.8 位于 `/usr/local/cuda-12.8`。
以 `KERNEL_ALIGN_FORCE_SM90=1` 编译 native 扩展后（`_EXT_AVAILABLE=True`，`fused_linear_logp_sm90` /
`_backward` / `_global_target` / `ordered_logp_merge` 符号齐备），现有 SM90 用例全部真实执行并通过：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export KERNEL_ALIGN_FORCE_SM90=1
export MAX_JOBS=16
python -m pip install -e . --no-build-isolation
python -m pytest tests/ops/logprob/test_linear_logp.py -k sm90 -v
# 24 passed, 29 deselected
```

覆盖：`test_sm90_forward_matches_native_bf16`、`test_sm90_strict_contract_is_bitwise_batch_invariant`、
`test_sm90_strict_contract_logits_return_and_padding_do_not_change_logp_bits`、
`test_sm90_strict_contract_backward_matches_logp_and_lse_reference`、
`test_sm90_rejects_out_of_range_target`、`test_sm90_falls_back_for_unsupported_inputs` 等正负路径。

**归属说明：** 这是 T01 bit-harness 到位前的**预验证**，用于提前暴露 actual kernel 问题；
WS1 Gate 要求的正式 bit-equivalence 判定仍以 T01 冻结的 P1 arithmetic spec + harness 为准（`#L173`、`#L700`）。
**注意**：M10 的 byte-exact 合同对象是 train 侧与 rollout 侧**同 kernel 同输入**的输出，
不是 kernel vs PyTorch golden（reduction 类 kernel 与 golden 的归约顺序不同，forward 本就无法 bitwise 相等）。

---

## 3. Canonical Sampling Record

### 3.1 记录结构

```text
canonical_sampling_record:
  record_version:   "p7-sampling-record-v1"
  case_id:          <stable case identity>
  engine:           megatron | miles_vllm | <producer>
  layers:
    requested:      <SamplingLayer>   # 调用方显式传入
    normalized:     <SamplingLayer>   # 框架归一化后（如 top_k=0 -> DISABLED）
    materialized:   <SamplingLayer>   # 落到 runtime 对象/tensor 的实参
    runtime_actual: <SamplingLayer>   # 实际执行时回读到的值（回读，不是配置值）
  canonical_fingerprint: <hash(规范化后的四层)>
  provenance:       <见 §6>
```

每个 `SamplingLayer`（合同 `#L211`）：

```text
SamplingLayer:
  temperature:  float
  top_k:        int | "DISABLED"
  top_p:        float | "IDENTITY"
  min_p:        float | "DISABLED"
  source:       config | api | framework_default | <producer>   # 该值从哪来
  raw:          <框架原始表达>                                    # 规范化前的值，用于溯源
```

### 3.2 Canonical sentinel（固定值）

一致性模式的 sentinel，任何一层规范化后都必须等于它（`#L210-L211`）：

| 参数 | canonical 值 | 允许的框架原始表达（先规范化再比较） |
| --- | --- | --- |
| `temperature` | `1.0` | 恰好 `1.0`；其它值 → mismatch |
| `top_k` | `DISABLED` | `None` / `0` / `-1` → `DISABLED`（规范化阶段无 vocab 上下文，不做 `>= vocab` 判定） |
| `top_p` | `IDENTITY` | `None` / `1.0` → `IDENTITY` |
| `min_p` | `DISABLED` | `None` / `0.0` → `DISABLED` |

**规范化规则（必须先规范化再比较，禁止直接比原始值）：**

- 数值比较用 canonical 后的语义相等，不是字符串相等。
- 框架默认值（未显式传入）也要被展开成显式字段，不能以"缺省 = 无所谓"放过。
- `top_k`/`top_p` 若被 producer 表达为 per-row tensor，则要求逐 row 全部为 sentinel。

### 3.3 四层校验规则

一致性模式下（`#L209`）：

1. 任一层 `temperature != 1.0` → `SAMPLING_PARAMS_MISMATCH`。
2. 任一层规范化后 `top_k != DISABLED` 或 `top_p != IDENTITY` 或 `min_p != DISABLED`
   → `SAMPLING_PARAMS_MISMATCH`。
3. 四层之间存在语义漂移（requested == sentinel 但 runtime_actual != sentinel，或反之）
   → `SAMPLING_PARAMS_MISMATCH`（这就是"实际语义漂移"）。
4. 任一层缺字段 / 无法回读 → `MISSING_PROVENANCE`。
5. 命中框架 fallback / 未声明 backend 分支 → `SILENT_FALLBACK`。

**关键约束：** `configured value` 不得冒充 `actual readback`（DoD `#L743`）。`runtime_actual`
必须来自执行侧回读，不能直接复用 `materialized`。

**实现诊断优先级**（fail-closed）：缺层 → `MISSING_PROVENANCE`；命中 fallback → `SILENT_FALLBACK`；
逐层 identity 不符（含 requested vs runtime_actual 漂移）→ `SAMPLING_PARAMS_MISMATCH`。
漂移不需要单独一条路径——任一层非 identity 即被逐层检查捕获，`failing_layer` 指出最早出问题的层。

---

## 4. M10 输入 Schema

按合同 `#L43-L61` 具体化。`[待 T01]` 处，dtype/shape 以 T01 manifest 冻结为准。

| 字段 | 类型 / 形状 | 必需 | 说明 |
| --- | --- | --- | --- |
| `case/run/engine identity` | immutable/versioned | 是 | 与 case_id 绑定 |
| `final_decoder_rows` | BF16 `[N, H]` | 是 | M10 唯一允许的 logits 来源 |
| `target_token_ids` | INT64 `[N]` | 是 | selected target |
| `active_token_mask` | BOOL `[N]` | 是 | 只比较 active action tokens |
| `lm_head_weight_local` | BF16 `[V_local, H]` | 是 | local vocab shard |
| `token/absolute layer` | canonical global identity | 是 | |
| `position/cache identity` | versioned state/page identity | 是 | |
| `checkpoint/weight_id` | immutable identity + fingerprint | 是 | |
| `failure_batch_context` | optional filter | 否 | run/checkpoint/sample/rank/module |
| `pre_update_weight_lock` | immutable fingerprint | 是 | 漂移即 case 无效 |
| `vocab ownership` | `[vocab_start, vocab_end)` | 是 | 无缺口、无重叠、无歧义 |
| `rank/group/order` | versioned metadata | TP 时是 | ordered rank merge 依赖 |
| `sampling requested/actual` | canonical record | 是 | 见 §3 |
| `Foundation ABI versions` | 5 类 ABI 版本号 | 是 | SemanticTensor/RoutePlan/ExchangePlan/ExpertBatch/CombinePlan |
| `boundary trace` | versioned TraceEnvelope | 是 | recorded 或 live |

### 4.1 M10 前置校验（在计算 logprob 之前）

- **vocab ownership 校验**：`[vocab_start, vocab_end)` 覆盖 `[0, global_vocab)` 无缺口/重叠
  （`#L29`、`#L719`）。不满足 → `INVALID_PROFILE` 或 `MISSING_PROVENANCE`（按 T01 manifest）。
- **selected target ownership**：每个 active target 恰好由一个 rank 拥有，owner count 必须为 1
  （`_validate_global_targets` / `owner_count`，`#L215`）。
- **padded vocab mask**：target 落到 padded 区（`patient >= global_vocab_size` 的部分）
  → fail-closed，不得当作有效 target（`#L215`）。
- **identity/profile gate 未过时停止数值归因**（`#L685`）。

---

## 5. M10 输出与数值合同

### 5.1 输出字段（合同 `#L65-L79` 的 T02 子集）

| 字段 | 说明 |
| --- | --- |
| `selected_logprob` | FP32 `[N]`，`z[t] - logsumexp(z)` |
| `selected_logprob_bytes` | active-token raw bytes（strict verdict 的唯一依据） |
| `selected_target_summary` | target value + global max/sum/LSE + ordered merge provenance |
| `actual_provenance` | backend/kernel/build/dtype/weight/shard/reduction/collective |
| `identity_discrete_gate` | exact gate verdict + failing field |
| `first_mismatch` | 失败时按 `(absolute_layer, sublayer_site, pass, event_index)` + canonical token/rank 键 |
| `result_cube` | T02 只填 case/engine/level 维度；boundary/config 维度归 T10/WS2 |

### 5.2 数值路径：复用现有 `linear_logp`

| 契约点 | 真实符号 / 位置 |
| --- | --- |
| 非 TP / TP 统一入口 | `tensor_parallel_linear_logp(hidden, lm_head_weight, target_ids, bias=None, *, tp_group, vocab_start_index=0, global_vocab_size=None)` |
| 非 TP 单卡 op | `NativeLinearLogpOp`（CPU/Triton-less 回退 + oracle） |
| local 统计（max/sum/target logit） | `_chunked_local_linear_logp_stats(...)` |
| ordered rank merge | `_merge_tp_local_logp(local_lse, local_target_logit, *, tp_group)`；CUDA FP32 走 `ordered_logp_merge` |
| target ownership 声明 | `_validate_global_targets` / `_assert_global_targets_async` |
| vocab shard 校验 | `_validate_tp_vocab_partition(_cached)` |
| `BN=64` 常量 | `SM90_N_SPLIT_CONTRACT == 64`（即合同的 `BN=64`） |
| 数值合同版本 | `SM90_LINEAR_LOGP_CONTRACT_VERSION = "cuda-fused-linear-logp-sm90-contract-v1"` |

**强制约束：**

- 必须调用上述现有实现；**禁止回退到任何未声明的 scorer**（`#L216`）。
- 一致性模式**不重放 sampling transforms**（`#L33`）。
- global max / sum / LSE 与 rank merge 顺序必须由 `linear_logp` 自身保证，T02 只**校验**
  ordered merge 的形态与 provenance，不另写归约（`#L29`、`#L215`）。
- M10 数值的 bit 合同由 P1 的 fixed-K/batch-invariant GEMM arithmetic spec 决定；T02 用
  T01 的 bit harness 验证 actual kernel，失败应归因 **actual kernel / spec mismatch**，
  不得在 P7 新写替代 kernel（`#L214`）。

### 5.3 LM Head LoRA 拒绝路径

- 检测到 LM Head LoRA（在 LM Head 权重路径上）→ strict 模式返回 `UNSUPPORTED_CAPABILITY`，
  **不得静默忽略**（`#L31`、`#L217`）。
- 注意区分：P5 expert LoRA **不在拒绝范围**，属于上游模型身份，必须记录并校验
  adapter/target/rank/scale（`#L32`）。实现里不要把两者混为一谈。

---

## 6. Provenance 字段（交付物之一）

`actual_provenance` 至少覆盖（合同 `#L69`、`#L35`）：

```text
backend           实际执行后端（cuda/rocm/ascend/triton/cpu）
kernel            实际 kernel / 实现符号名
kernel_contract   如 SM90_LINEAR_LOGP_CONTRACT_VERSION
build             扩展 / 库 build identity
dtype             hidden / weight / 累加 / 输出
weight_id         权重 fingerprint（含 pre_update_weight_lock）
vocab_shard       [vocab_start, vocab_end) + world_size
reduction         BN 常量 + rank merge 顺序
collective        TP group / ordered merge collective（如适用）
fallback_used     是否命中 fallback / 未声明 backend
```

规则：缺失或 silent fallback **不能判定为通过**（`#L35`、`#L683`）。

---

## 7. Gate 决策流程与 Stub 设计

### 7.1 决策顺序（fail-fast，在计算/比较之前）

```text
1. identity / discrete profile gate        -> INVALID_PROFILE / IDENTITY_DRIFT
2. sampling params gate（§3.3）            -> SAMPLING_PARAMS_MISMATCH / MISSING_PROVENANCE / SILENT_FALLBACK
3. LM Head LoRA detect                     -> UNSUPPORTED_CAPABILITY
4. M10 前置校验（§4.1）                     -> MISSING_PROVENANCE / INVALID_PROFILE
5. 调用现有 linear_logp 计算 selected_logprob
6. 输出 selected_logprob_bytes 供 T09 做 byte verdict；诊断指标不参与 verdict
```

任一步失败即返回，交给 T09/T10/T11，不继续下游归因（`#L683-L687`）。
non-finite / zero-active 直接 fail-closed，不进 tolerance comparator（`#L459`、`#L689`）。

### 7.2 Stub / synthetic fixture（合同 `#L601`、`#L641-L643` 允许的先做）

在 T01 start kit 未到位前，用自建 stub 冻结自己的接口与测试：

| Stub 组件 | 内容 | 约束 |
| --- | --- | --- |
| synthetic fixture | 固定 seed 生成 §4 全字段（`final_decoder_rows` BF16 `[N,H]`、`target_token_ids` INT64 `[N]`、`active_token_mask`、`lm_head_weight_local` BF16 `[V_local,H]`、vocab ownership、canonical sampling record） | 必须是合成/脱敏数据，禁真实 checkpoint/token（`#L769`） |
| stub provider | 只实现接口，直接返回 fixture，不做真实推理/训练 | 不替代数值实现 |
| CPU oracle | 复用 `NativeLinearLogpOp` 作为正确性 oracle | 不新写 oracle |
| stub sampling producer | 造 requested/normalized/materialized/runtime_actual 四层，含负向样本 | runtime_actual 必须是"回读"语义，不能直接抄 materialized |

**关键：live replacement ABI 冻结。** stub 与真 provider 必须同 schema/签名；未来替换后
**case ID、ABI、comparison policy、runner、artifact、验收命令保持不变**（`#L641-L643`）。
stub 阶段就要把 case ID 生成规则定死，否则 WS2/Integration 无法复用。

**stub 的边界：** 只能喂数据、占位接口；**不允许**为了跑通而自写 `linear_logp` 或 sampling scorer。

---

## 8. 测试计划

### 8.1 正向

- canonical sentinel record（`temperature=1.0 / top_k=DISABLED / top_p=IDENTITY / min_p=DISABLED`）
  四层一致 → PASS。
- 非 identity 表达但可规范化到 sentinel（`top_k=0`、`top_p=None`、`min_p=0.0`）→ PASS。
- M10：单卡与 TP 路径调用现有 `linear_logp`，对比 CPU oracle；用 T01 bit harness 验证 actual kernel。
- `selected_logprob_bytes` 稳定（同 case 重复运行 byte-identical）。

### 8.2 负向（全部 fail-closed，`#L219`）

| 场景 | 期望状态 |
| --- | --- |
| `temperature != 1.0` | `SAMPLING_PARAMS_MISMATCH` |
| `top_k` 启用（非 DISABLED） | `SAMPLING_PARAMS_MISMATCH` |
| `top_p != IDENTITY` | `SAMPLING_PARAMS_MISMATCH` |
| `min_p` 启用 | `SAMPLING_PARAMS_MISMATCH` |
| requested=identity 但 runtime_actual 非 identity（语义漂移） | `SAMPLING_PARAMS_MISMATCH` |
| 任一层缺字段 / 无法回读 | `MISSING_PROVENANCE` |
| 命中 fallback / 未声明 backend | `SILENT_FALLBACK` |
| LM Head LoRA 存在 | `UNSUPPORTED_CAPABILITY` |
| target 落在 padded vocab 区 | fail-closed（按 manifest） |
| vocab ownership 缺口/重叠/歧义 | `INVALID_PROFILE` / `MISSING_PROVENANCE` |
| pre_update_weight_lock 漂移 | `PRE_UPDATE_WEIGHT_DRIFT` |
| non-finite / zero-active | `NON_FINITE` / `ZERO_ACTIVE_TOKENS` |

### 8.3 WS2 延续（实现留 WS2，字段本 Spec 冻结）

- 逐 rank canonical sampling record 完整且一致（`#L218`）。
- `actual_provenance` 每个 expected rank 齐全；缺 rank → `MISSING_RANK`。
- 同 global token 可跨并行配置唯一映射；歧义 → `AMBIGUOUS_GLOBAL_TOKEN_MAPPING`。

---

## 9. 开放项 / 待 T01 确认（可能的阻塞）

1. **T01 冻结的代码基线 commit**：当前用 `origin/main @ 4406afc`；若 T01 钉了具体 commit，需 rebase。
2. **M10 输入/输出 schema 的正式 dtype/shape/layout**（本 Spec 按合同 `#L43-L80` 推断）。
3. **boundary-contract manifest 的 M10 行**：`comparison_mode`、`tolerance_contract_version`、`fixture_checksum`。
4. **canonical hash version**：`boundary_hash_index` 用。
5. **P1 GEMM arithmetic spec + bit harness**：验证 actual `linear_logp` kernel 依赖它（`#L173`）。
6. **T01 provider stub / synthetic fixture / CPU oracle** 是否已有可复用实现。

以上任一未决且无法用本 Spec §7 自建 stub 顶替时，再向 vensen 确认。

---

## 10. 交付物与验收（对齐合同）

**交付物**（`#L221`）：gate、参数正规化逻辑、provenance 字段、正负测试。

**验收**（`#L223`）：任意非 identity sampling 设置、实际语义漂移、LM Head LoRA 或未声明 backend
都在**计算/比较前**明确失败，不能静默继续。

**WS1 Gate 关条目**（`#L701`）：T02 sampling/M10 contract 通过全部正向和负向测试。

**共同 DoD**（`#L739-L750`）：configured 不冒充 actual readback；schema/hash/artifact 带版本；
复用指定基础设施；Owner 提供可执行命令 + 结果摘要；Reviewer（T09）批准。

---

## 11. 当前状态与待确认 / 阻塞项

> 快照：2026-10-14。分支 `feat/p7-t02-sampling-m10-contract`（基于 `origin/main @ 4406afc`）。

### 11.1 已完成（附证据）

| 交付项 | 位置 | 证据 |
| --- | --- | --- |
| canonical sampling record（四层 + 版本化 fingerprint） | `rl_engine/validation/sampling_params/schema.py` | `tests/validation/sampling_params/test_sampling_gate.py` |
| 参数正规化 + fail-closed gate | `rl_engine/validation/sampling_params/gate.py` | 同上（缺层 / fallback / 非 identity 负向） |
| M10 输入 schema + 前置校验链 | `rl_engine/validation/m10/{schema,precheck}.py` | `tests/validation/m10/test_m10_precheck.py` |
| synthetic fixture + stub provider（冻结 read ABI） | `rl_engine/validation/m10/stub.py` | 同上 |
| M10 计算片（复用 strict `linear_logp` + raw bytes + provenance） | `rl_engine/validation/m10/score.py` | `tests/validation/m10/test_m10_score.py`（4 项 SM90-gated 真跑） |
| H20/SM90 预验证证据 | 见 §2.1 | `pytest tests/ops/logprob/test_linear_logp.py -k sm90` → 24 passed |

```bash
python -m pytest tests/validation/sampling_params tests/validation/m10 -q   # 64 passed
```

回归：`pytest tests/validation tests/ops/logprob -q` → **687 passed / 3 failed / 20 skipped**。
3 个失败均与本任务代码无关：2 个是全量跑时 5s scoring 超时的 flaky（单跑即过），
1 个是扩展编译缺 `KERNEL_ALIGN_DET_GEMM_SM90=1`（见 §11.4）。

### 11.2 未完成（按共同 DoD `#L739-L750`）

- [ ] **简短使用说明**：DoD `#L40`/`#L747` 要求代码、单测、负向测试和文档同时提交。
- [ ] **Reviewer 批准**：T02 的 reviewer 是 **T09**，尚未评审。
- [ ] **WS2 延续**：逐 rank canonical sampling record 完整/一致校验（`#L22`/`#L218`；WS1 完成后同一 Owner 自动延续，非可选）。
- [ ] **PR**：本轮只 push 分支、**不开 PR**；分支已推到 fork `Endeavor-19/RL-Kernel`（origin 无写权限，见 §11.4）；PR base（`test` 还是 `main`）待确认。

### 11.3 待确认 / 阻塞（[待 T01]）

详见 §9，另：

- [ ] **正式 bit-equivalence 判定**必须用 T01 的 P1 fixed-K/batch-invariant GEMM arithmetic spec + bit harness（`#L173`/`#L700`）；本任务只做了 §2.1 的预验证，归属已在 §2.1 标注。
- [ ] **boundary 声明字段**（`boundary_phase/sublayer_site/event_index/branch_id/predecessor_event_ids`、`comparison_mode`、`producer_verdict`、`fixture_checksum`）由 T01 的 boundary-contract manifest 提供，T02 只消费。
- [ ] **identity/discrete profile gate** 是第一条决策步，归 **T03**，不在 T02 范围。

### 11.4 环境已知问题（非代码缺陷）

- **网络**：本机 `github.com` 直连不稳定（`curl` 超时、`git ls-remote` 间歇失败），git/gh 操作前需
  `source /etc/network_turbo`（AutoDL 学术加速，代理 `http://10.0.0.11:12798`，仅加速
  github/huggingface，不承诺稳定）。`api.github.com` 通常可直连。
- **写权限**：`Endeavor-19` 对该仓库只有只读权限（token scopes 含 `repo`，但
  `permissions = {push: false, pull: true}`），直接 push `origin` 返回 403。分支已推到 fork
  `Endeavor-19/RL-Kernel`；本地 `fork` remote 指向该 fork，分支 upstream 已修正。
  拿到写权限后可迁回 `origin`（PR base 仍待确认）。
- **凭据**：本机 gh 为 **2.4.0**，**没有 `gh auth token` 子命令**（误用会得到 usage 文本而非 token）；
  token 改从 `~/.config/gh/hosts.yml` 读取，并以一次性 `credential.helper` 注入——
  未写入 remote URL、未落盘、未改 git config。
- 本机扩展以 `KERNEL_ALIGN_FORCE_SM90=1` 编译，**未开** `KERNEL_ALIGN_DET_GEMM_SM90=1`，导致
  `tests/validation/operators/test_ws1_candidate_evidence.py` 有 1 项失败（`refusing naive fallback`）；
  补上该 flag 重编即可消除。
- `pip install -e .` 会连带改动 `torch_env` 依赖（transformers/typing-extensions 等）；仅需扩展时用 `--no-deps`。
- 本机单卡：TP 多 rank 实跑不可用，WS2 需多卡环境。
