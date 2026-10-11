# Rule2Region 原主线核对与候选支线保存

核对固定原始发布提交 `cf22ddd8e46ee83ddd9565b058e1df4e362377ea`，恢复基线 `74f6ad2b40afe8dda44f327037253957daaea8a0`，本轮起点 `5d9554479e2e66c966d0272f7a04a7a1d0c291e0`。`SOURCE_EVIDENCE.json` 固定已读文件字节哈希与 Git blob。没有按最近对话重写根 README、旧任务、冻结预测或 ROI。

## 原文与贡献边界

原始发布提交 `README.md`：

> 业务规则 → 候选监控 ROI 与配置。研究目标是让 LLM 根据规则和固定摄像机场景提出可审查的区域建议，由人确认后交给固定检测器执行。

> 计划中的主链路为：业务规则与场景输入 → LLM 候选 ROI/配置 → 几何与规则校验 → 人工确认 → 固定检测器执行与日志。LLM 建议不得直接变为上线配置。初期不以更换或训练检测器作为主要研究变量。

> 下一步是冻结业务规则与人工确认接口，构建可追溯 ROI 候选合同，在独立数据上检验区域覆盖、配置时间、误报漏报和拒绝建议边界。

上述原文在恢复基线仍保留。消防通道杂物、室内吸烟、工作区手机使用是目标场景，未验证其检测能力。这里是预期研究贡献，不是已有创新或效果结论。

恢复基线 `docs/research-handoff/tasks/status-20261009.json`：

> R2 is a screening ROI, not a physical floor mask; R1 and R3 retain unknown_scope.

恢复基线 `docs/research-handoff/tasks/status-20261010-boundary16.json` 的下一项：

> Review frozen ROI pixel geometry and exhaustively annotate all visible people in all 16 original full frames, including detector omissions and image-of-person negatives; then compute preregistered metrics. Preserve dev exposure and do not tune ROI to these predictions.

`docs/research-handoff/reports/progress-20261010-boundary16.md` 固定的 F/H/M 比较共享同一全图检测缓存、ROI 与阈值，正面积跨界框完整保留；独立测试镜头族为 0，7 项 GT 指标仍 null，不能用不同覆盖集合的候选总数当效果。完整原始私有冻结协议/执行包只公开身份元数据，本次不能声称已恢复其全部字节；上述是已验证公开状态与冻结约束。

## 方向变化的证据

原 README 的主目标不包含“训练一个门口进出事件判别器”。门口事件在 `research/continuous-cache-20261010/README.md` 是缓存、几何、轨迹/方向与事件证据可用性的开发验证用例；四臂全部 unknown，未建立识别收益。仅凭这些已定材料，不能反推进出事件是原项目主目标。

近期另选 ChokePoint、讨论物理内外/连续边界/事件 GT，并提出统一多判据决策训练层，研究输入输出已从“规则与场景→审查 ROI/配置”转向“视频证据→事件决策”，评价单位也转向事件准确率/效率。新方向可作为独立候选支线，价值需另行文献分析；本轮不决定架构或训练，不改原主线。阈值、方向、unknown 边界及 AI≠人工真值的旧约束继续保留。

## 当前素材核查及停止点

预先冻结条件后抽查已下载 C1 的六段、30 张原图，原始成员字节全部验证，0 个可纳入的可见物理进出候选；不是全 2292 图穷尽结论，不是真实事件数为 0。A/B 保持物理侧别 unknown 的边界/排除记录。所有媒体及帧级 AI 观察私留，不新增推理/训练/下载；后续素材实验暂停。详见 `../aligned-visible-event-admission-20261011/RECEIPT.json`。

## 回归主线最近的一项可执行工作

先对原 16 图冻结任务做“待恢复原件/几何/人审记录”逐项对照清单，绑定现有公开输入身份、预测身份、F/H/M 覆盖和原归档 SHA，列出每图 M0 几何与全图人员/照片负例的未审状态。只做合同与缺口核验，不用 ChokePoint 替换原输入、不从测试期望重造预测/真值，也不重新询问已确认的 C 室内业务范围。现有公开 evidence/tasks 足够推进这一清单；实际区域覆盖、误报漏报与抑制指标仍须原件和人工标注，暂不可计算。

拿到授权原件后的原既定下一步仍是完整 16 图人审及冻结几何复核，在相同固定检测缓存上评估 F/H/M，记录候选暴露；当前缺原件不转去扩进出事件框架。若要研究通用规则到 ROI 的贡献，依据根 README 另外冻结可追溯候选合同与人工确认接口，并在独立视野测试区域覆盖、配置时间及拒绝边界，不能声称既有人员计数代码已实现通用 ROI 生成。
