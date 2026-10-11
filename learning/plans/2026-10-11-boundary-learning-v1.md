# 门与区域关系学习计划 v1

**版本：** boundary-learning-plan-v1，2026-10-11 UTC。用户已批准保存并开始初期实施。
**Goal：** 学习允许画面/场景信息中的门与通道实例、两侧区域关系、跨视角同门对应与证据充分性，供完整可信拓扑上的确定性 cut 使用。
**Architecture：** 先现成基线，再冻结视觉编码器与候选关系头，按真实错误决定 LoRA/蒸馏。输入、预标注、人工确认真值独立；不把跨场景区域 ID 当分类目标。
**Tech Stack：** Python 标准库合同校验；可选本地 PyTorch，当前环境未安装，不安装依赖。
**Spec：** 本文件保存 2026-10-11 用户授权任务的研究规格；公开已读 `README.md`、`docs/external-assets.md`、`research_tools/roi_structure_protocol.py` 仅作为旧工程约束。

## 约束与实际上下文

本分支仅新增 `learning/`；基点为公开 main `cf22ddd`。公开 main 快照未包含项目简报、`.agents/skills` 或私有冻结16图材料。启动时 pilot 分支尚未在远端出现；发布前只读获取了 `task/building-boundary-pilot-20261011` 的 `850b4f3f3f9fc1ac955b16eaa070493a2e453b4d`，阅读其 `pilots/building_boundary/PROTOCOL.md`、`OFFICIAL_ADMISSION.md` 与 `pilots/building_boundary_v2/PROTOCOL_V2.md`（用户给定早期基线 SHA `40f0240b6929c13c6ce9bad62b25a5111ec49e85`）。不据此推断它们不存在于父任务；不替换原 main 冻结研究，不编辑另一任务工作树，不重复几何/初值修复。站点由唯一作者负责，本分支仅交付合同和待获取清单。运行位置为云执行环境 `/workspace/Rule2Region-learning`，不是 Mac；父任务负责重要成果回存 Mac。

不下载大型权重/视频包、不长训、不调用付费 API、不配置凭证、不上传原媒体或日志；允许独立分支提交推送公开计划、代码、自制软件测试数据。没有训练预算、样本量或模型性能实测，不编造这些值。

视觉画质和时序错位不属于本阶段研究变量；材料中若确有这些问题只登记为输入限制。清晰同步也可能缺乏楼层/室外/隐藏通道身份，应保留 unknown；不得凭外观补齐拓扑。

## 研究问题与任务

1. 给定画面和允许场景信息，能否定位门/通道实例并关联当前场景候选区域对？候选漏检单独衡量；第一版关系头消费候选，不实现候选检测器。
2. 两侧区域是否相连、不同视角是否同一物理门？输出 connected/not_connected/unknown、same/different/unknown，而不是固定区域名类别。
3. 证据是否足以判定？人工 unknown 是一个语义标签；模型低置信拒判是经 calibration 选择的动作，两者单独统计。
4. 完整可信图 G=(V,E) 中目标区域集合 S 的边界为 δ(S)={e=(u,v): exactly one endpoint in S}。房间、楼层、整栋用不同 S；同一内部门可属于房间边界而不属于整栋边界，楼梯/电梯/连廊不能忽略。“外门”分类不等于所有统计边界。
5. 仅当目标相关完整性清单、每条边身份/两侧关系与跨视角唯一性通过人工核验，cut 才可进入几何投影与固定计数器准入。未知/未观测连接阻断完整边界结论；无完整性证据的部分图不能算完整覆盖。几何、摄像头重复归属、初始人数/重置等由现有 pilot 工作负责。

## 对照与评估冻结

同一输入包、允许信息、候选生成版本、人员检测跟踪器、投影/准入规则、目标 S 和计数初值条件；规则不能读更多图纸，模型不能读取独立 GT。比较：规则基线；未微调 SmolVLM2 / InternVL3；冻结编码器关系头；仅在错误支持时的 LoRA。Laya 可对同一输入导出的文本 JSON 做辅助决策基线，不作为视觉编码器，人工提供正确图不能混入它的输入。完整图 oracle 只给上限诊断，独立报告，不列为同输入模型。

阶段锁定 train/calibration/test 的 building/layout-family 联合分组，物理门、源图纸哈希、同图变体不得跨组；所有 augmentation 在分割后派生。校准集选择拒判阈值、超参；test 只在版本冻结后一次评估。公开预训练污染不可彻底排除，登记来源及风险，优先额外授权独立楼宇。真实 building/layout family 不足以三分时，暂停跨楼验证，不随机拆帧充数。

报告：门/通道候选召回及定位；两侧关系准确率及 unknown；同门对应 precision/recall；δ(S) 边 precision/recall、每目标完整覆盖成功率（GT 必须完整）、额外边/漏边；拒判风险-覆盖曲线与已知/unknown 分层、错误接受 unknown；固定计数器事件误计/漏计、人数漂移 MAE/最大绝对误差/最终偏差（确认初值与参考人数独立）、配置/复核时间与修改次数。误差按楼宇而非帧 bootstrap，样本不足只报描述性值。不用 accepted accuracy 隐藏拒判或漏候选，单独报真值未知比例及全量分母。

## 数据来源与许可

官方核对日期 2026-10-11；链接和获取门禁见 `learning/data/acquisition_checklist.json`。CAVIAR 的店门/走廊与 ChokePoint 门口多视角仅用于真实子任务，不是整楼 GT；ProcTHOR 是合成布局，需另外渲染相机和人流，仿真不能冒充真实楼宇。未找到现成真实整楼完整门—区域—事件—人数真值包。本阶段真实样本 manifest 为空，绝不用虚构图片替代。

本站点待标注包需提供：合法许可的场景视图/相机身份；允许模型看见的场景说明或图纸；独立实际区域与物理门 ID；楼宇/布局族/源图纸/变体身份；现场或图纸证据；目标房间/楼层/整栋集合；所有出入口和未见连接的完整性核验。人数评估再补带独立核验的事件/参考人数/初始人数材料，不在此重复计数器准入实现。AI 预标注永不自动成为人工真值。

## 实施任务和验收

### P0：合同与站点交接（本轮）

- [x] 保存本计划；建立输入/独立真值/AI/复核 schema、空真实 manifest、自制软件 fixture。
- [x] 实现 `learning.contract.validate_manifest(payload)`，拒绝泄漏、无依据人工确认、版本不匹配、跨 split building/layout/door/图纸/变体、非法标签或缺失真值；`python -m learning.validate ...` 可独立运行。
- [x] 交付 `learning/annotation-guide.md` 与公开待获取清单；真实媒体不上传；schema 软件检查不能证明现场 GT 正确。
- [x] 验收：合同正反例通过；没有真实样本就保持空列表。停止：来源/许可/完整性不明则不获取或不称完整 GT。

### P1：最小可训练软件（本轮）

- [x] `learning.model` 冻结编码器接口、门+区域对关系评分、视图门对评分及充分性输出；`learning.dataset` 只把输入特征给模型，独立人工 GT 才成为 targets。
- [x] `learning.train` / `learning.evaluate` 消费本地可追溯冻结特征；软件模式只消费显式标记的合成张量。无权重 smoke 只验证 shape/梯度/冻结/分割/泄漏，不叫模型实验。
- [x] 标准库测试实际运行；PyTorch 缺失则 skip 并明确记录。正式训练入口不能接受合成 fixture；未确认记录不能充当监督样本。
- [ ] 停止：依赖缺失只静态验证，不自动装包/购算力；不生成模型指标或训练预算。

### P2：真实预标注和复核（材料到位后）

- [ ] 父任务取得最小合法真实包与楼宇/布局族身份。AI 先预标注，高风险/unknown 置顶；普通建议默认采用只代表待复核状态，人工确认/修正带证据及三版本进入历史。
- [ ] 每个关系独立标注；不能从“看到门”推断另一侧区域。证据不足标 unknown，缺证据属于未标注而不是负例。
- [ ] 验收：独立复核及冲突裁决记录完备，校验实际许可/媒体哈希；冻结身份连接组件 split。当前没有数据，AI 预标注尚未执行。
- [ ] 停止：GT 无证据、building 身份不可靠、标注分歧无法裁决；索取最小补充视角/平面图/现场核验，保持 unknown。

### P3：基线→关系头（后续需数据和执行条件）

- [ ] 先现成同输入基线，记录权重 revision、预处理、提示和本地资源小测；不凭官方通用 benchmark 推断本任务效果。
- [ ] 仅人工审核 train 标签监督关系头，校准集选择阈值。记录训练样本量、耗时/显存/能耗（可测时）、迭代次数、种子与实际支出，预算先实测后授权。
- [ ] 验收：冻结后盲 test，报告所有上述指标及按楼宇置信区间；可接受风险上限由用户在校准前填写并版本化，未填写不得宣称上线达标。
- [ ] 停止：完整边界/unknown 风险不满足上限、跨楼无提升或成本无收益，不升级训练规模。

### P4：条件性 LoRA / 蒸馏

- [ ] 只在错误分析显示冻结表征不足且合法标签/资源可用时提出具体小试；同 split/同输入/同评估。teacher 预测仍是预标注，test 不用 teacher 作为真值。
- [ ] 验收：相对关系头边界覆盖、风险和成本收益得到独立 test 支持；模型同门/图纸记忆或拒判退化则停止。

## Review Focus

隐藏通道导致完整性错误；局部区域 ID 重命名应不改变输出；同门跨视角跨 split；GT/审核理由漏进模型输入；人工 unknown 与置信拒判混同。前四项由合同/软件测试或后续完整性准入核验，最后一项由评估输出区分。本轮不实现完整拓扑 cut 和几何准入，不声称已解决整栋人数统计。

## 算力补充（同日用户指导）

默认后续资源基线 Kaggle 2×T4，每卡16GB；DDP 每进程模型副本，不池化显存。优先缓存冻结视觉特征，关系头先单卡少量步 profile 峰值显存/step time/loss，再决定双卡 DDP 或并行种子。500M–1B 小 VLM LoRA 次选，3B QLoRA 仅后续实测，不保证可装入。checkpoint/resume、固定种子/manifest、test 禁调阈值见 `learning/README.md`。当前账户配额/时限未知，不访问凭证、不启动作业、不付费。官方依据：[NVIDIA T4](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)、[PyTorch DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)、[PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)。

与已读 pilot v2 的衔接：楼梯/电梯在途必须明确独立区域或归属；关系学习只提供候选证据，不能绕过几何侧锚点、配置 SHA 审批和初值会话资格。v1/v2 合成零误差不作为本学习任务的模型结果或真实 GT。上述协议只读，本分支不复制或修改 pilot 实现。
