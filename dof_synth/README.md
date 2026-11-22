# 后期景深合成器（Post-process Depth-of-Field Compositor）

输入带已知相机参数的**颜色图 + 米制深度图**，依据焦距、焦平面和光圈用薄透镜
模型计算圆弥散（Circle of Confusion, CoC），并通过两种物理上可对照的分层
方法真实合成前景/背景虚化，而不是对整幅图套一层模糊滤镜。

## 光学与单位约定（全项目一致）

| 物理量 | 单位 |
| --- | --- |
| 场景距离 / 深度图 | 米 (m) |
| 焦距、传感器尺寸、光圈直径 | 毫米 (mm) |
| 传感器上的 CoC 直径 | 毫米 (mm) |
| 像素上的 CoC | 像素（`signed_coc_px` 为**直径**，`radius_px` 为半径） |

投影约定：相机在原点沿 +Z 观察；深度是针孔中心射线交点的轴向距离；
像素中心位于 `(i+0.5, j+0.5)`。

薄透镜 CoC（`camera.py`）：

```
c(z) = D f (S - z) / (z (S - f)),    D = f / N
```

符号约定：`c > 0` 前景（焦平面前），`c < 0` 背景，`c = 0` 恰在焦平面。
同时提供超焦距、景深近/远界、入瞳直径与像素间距的换算。

## 两种求解方法

1. **前向散射 `scatter.py`（从近层向远层散射光斑）**
   每个表面像素把接收到的光按自身 CoC 圆盘散射；按深度从近到远，用预乘
   alpha 的 over 合成逐层累积。薄前景即使只占几个针孔像素，也会在它覆盖的
   每个输出像素沉积完整光斑能量；背景光被前景覆盖率按 `1 - a_fg` 衰减，
   因此不会穿透不透明前景。

2. **反向分层取样 `gather.py`（按输出像素反向分层取样）**
   每个量化深度层用该层 CoC 圆盘做卷积，输出像素前向到后向地合成。前景层
   累积覆盖率并据此遮挡后续背景层；背景只能填补前景与更早背景层留下的透明
   部分。两方法共用同一光学核与掩码，结果应在分层量化误差内一致（测试断言）。

两者都基于 FFT 圆盘卷积（`kernels.py`），圆盘核归一化为和 1，从而在图像
内部严格守恒能量（仅图像边界有真实的越界损失）。

## 遮挡、显露与边缘污染（显式声明，不隐藏）

- `geometry.py` 检测深度不连续与给定细结构轮廓，**只标记轮廓边界像素**为
  边缘深度污染；表面内部保留实测深度。
- `inpaint.py` 计算前景散景足迹下针孔看不到的**未知背景** `unknown_background`，
  并区分：
  - 可达区域：在声明半径内存在可见背景样本，用最近邻外推填充；
  - **不可恢复区域** `unrecoverable`：超出声明半径、没有任何输入证据，
    仍给出填充值以保持合成有定义，但明确掩码标注其为近似。
  - 填充只允许从**可见背景**取深度，绝不会用遮挡它的前景深度，否则会给显露
    区域错误地套上巨大的前景 CoC。
- 原始 `color` / `depth` 永不被修改，移动焦平面始终从原始测量重新计算。

## 光学参考与验证

`reference.py` 用蒙特卡洛薄透镜光线追踪（在入瞳圆盘上均匀采样）渲染同一场景，
作为与实现无关的对照真值；配套的针孔通道产出两种方法消费的颜色/深度对，
保证相机约定完全一致。

`scene.py` 的程序化场景专门压测失效模式：

- 仅数像素宽、6 cm 深的**细前景板**（带解析轮廓掩码）；
- 分列其两侧、散景可触及的**亮点背景**（测量光斑能量与光晕）；
- 多个不同深度的哑光层与近/中景块。

`metrics.py` 量化四类问题：亮点总通量守恒、细板遮挡侧光晕（对照光学参考的
超出量）、前景足迹内穿透误差、相对参考与两方法间的 RMSE，并附带一个故意朴素的
“逐像素半径模糊”基线用于对照。

## 遮挡重影诊断（边界追踪 + 内核证据链）

遮挡边缘出现重复轮廓时，诊断必须证明原因，而不是靠裁掉边界“改善”指标。
`diagnostics.py` 提供以下实现，全部由**真实求解循环**产出（`trace.py` 在
`scatter` / `gather` 内部记录，绝不另行重写一份会漂移的模拟）：

- **景深边界追踪** `track_dof_boundaries`：逐像素找出远近深度不连续处，记录
  有符号 CoC、所在半径层与前后景分层。遇到非有限/非正深度**直接报错**，绝不
  用默认值掩盖（`defaults_used=False`）；始终统计 `included_full_frame=True`
  的全幅指标。
- **分层碎裂检测** `layer_fragmentation` / `depth_continuous_components`：
  用并查集把 `|Δdepth| ≤ step` 的相邻像素归为同一物理表面，再检查该表面是否
  被量化拆进多个半径层。这正是“同一表面被当成互相遮挡的两层 → 接缝覆盖率
  空洞 → 背景外漏成重复轮廓”的根因。
- **有效样本来源**（provenance）：background / foreground / edge /
  unknown_fill / unrecoverable，随深度与颜色一起进入内核追踪。
- **重影像素反指近远层贡献** `attribute_pixels`：把每个追踪像素的每一次层贡献
  （覆盖率、权重、能量、该层前后景符号与来源）展开，可逐项复算出输出像素。
- **阶段定位** `localize_stage`：把异常归到最早的违约阶段
  （input → prepare → quantize → layer_assign → convolve → composite → output）。
- **重影宽度与样本归属** `measure_ghost`：在全幅上测量重复轮廓宽度、峰值与
  RMSE，并**单独**报告边界带指标；`cropped=False`，带宽指标是子集而非替代。
- **已知深度阶跃实验** `known_depth_step_probe`：构造平坦阶跃、后退侧面、
  深度噪声三种已知情形，并**调换深度相同两层的输入序**（`layer_override` +
  `layer_order`）——顺序不变（`order_swap.invariant=True`）即排除“归一/合成
  顺序”错误；再用 `side_face` 的 scatter 回弹显著大于 gather 把原因锁定到分层
  碎裂，而非深度噪声或核覆盖。
- **日志成本受控**：追踪像素数、每像素事件数、总事件数都有上限，命中即累加
  `dropped_events` 并置 `truncated`；“停止记录”与“本来就没有”不会被混淆。

### 复核结论（可复现证据）

在 256 px 验证场景上，细板可见侧面沿深度从约 0.700 m 退到约 0.743 m，聚焦
处的 CoC 对深度敏感，使**同一物理表面**跨进不同半径层。over 合成在接缝处
留下覆盖率空洞，背景从外侧透出，形成宽约 2 px、峰值约 +0.024/+0.039 的重复
亮轮廓（对照薄透镜参考）。诊断与已知阶跃实验一致给出：

```
cause = layer_fragmentation     stage = layer_assign
composite 阶段同时记录覆盖率空洞   （直接症状，而非根因）
order_swap.invariant = True     （排除归一 / 合成顺序）
depth_noise 实验不改判            （排除深度噪声）
```

运行 `scripts/run_validation.py` 会把完整证据链与 `evidence_digest` 写入
`metrics.json`；同一输入重跑，摘要**逐字节一致**（`后期景深复跑证据保持一致`）。

## 交互界面（与合成画布联动）

界面由后期景深内核直接驱动：拖动焦点/光圈即在原始测量上重跑求解器，诊断面板
同步刷新。除了合成图与各类掩码，还提供 **occlusion diagnosis** 视图：在合成
画布上把分层碎裂足迹按原因着色、框出被追踪的重影像素；右侧面板显示每层散焦
（半径/代表深度/前后景符号/样本来源）以及逐像素的样本归属表。

启动方式（任选其一）：

```bash
python3 -m dof_synth.app      # 程序化验证场景，推荐
python3 scripts/run_ui.py     # 等价入口
```

- 拖动 **focus (m)** 滑块，或在合成图上点击把焦平面吸附到该像素深度；
- 调整 **f-number** 观察分层散焦与遮挡边缘如何变化；
- **scatter / gather** 切换两种求解方法；
- 在掩码列表选 **occlusion diagnosis**，再点 **Diagnose occlusion**：界面用
  `track_dof_boundaries` 挑出最强内部遮挡边缘，调用带 `TraceRequest` 的真实
  求解，然后经 `diagnose_occlusion` 显示 cause/stage、重影宽度与样本归属；
  若没有可用边缘或追踪缺失，界面**显式报错**（“No occlusion edge”/“Diagnosis
  failed”），绝不填入默认结论；
- **Save output + masks** 保存当前 16-bit PNG 与掩码 npz。

## 运行

依赖安装（无 root 时用用户态 pip）：

```bash
python3 -m pip install --user --break-system-packages numpy scipy matplotlib pytest
```

端到端验证（写出实际像素 PNG、掩码与 `metrics.json`、`raw_inputs_and_outputs.npz`）：

```bash
MPLBACKEND=Agg python3 scripts/run_validation.py --size 256 --lens-samples 128 --layers 64
```

交互界面（拖动焦平面滑块、或在图像上点击把焦平面吸附到该像素深度；切换光圈、
两种方法与各类掩码；联动的遮挡诊断面板；保存输出）——详见上文“交互界面”一节，
启动命令为：

```bash
python3 -m dof_synth.app   # 或 scripts/run_ui.py
```

测试：

```bash
MPLBACKEND=Agg python3 -m pytest tests/ -q
```

## 输出内容（`outputs/`）

针孔颜色/深度、薄透镜参考、scatter、gather、朴素基线、误差图、有符号 CoC、
边缘/前景/未知/不可恢复/焦内/细结构真值/亮点掩码，以及包含原始颜色与深度的
npz（可随时用新焦平面重新计算）。
