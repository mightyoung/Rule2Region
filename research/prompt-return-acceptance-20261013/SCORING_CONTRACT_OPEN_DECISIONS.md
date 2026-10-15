# 正式评分合同待决草案（非新协议、非执行许可）

盘点基线：`9881cb1f679f4c5c8edef588560bf76f9b91bf02`。仅整理已有门禁与待决项，不实现评分器，不改冻结v2/v3、不产生分数或GT。这里的“global one-to-one”只指未来独立评估器的匹配唯一性要求，**还不是已定义算法**。

## 已有约束与未定规则

| 维度 | 已有实现/已确认范围 | 尚未冻结，不能推定 |
|---|---|---|
| 匹配对象 | `target_metrics(labels,predictions,routing)`声明须有独立验证的global one-to-one评估器；四个目标指标名已列出 | 是逐帧person bbox↔完整人工对象集，还是门/关系/事件；global的图像、相机、类别、时间作用域；ignore/遮挡/重复预测规则 |
| 几何与阈值 | ROI路由使用原图连续边坐标、零几何容差；无效/空ROI走review_unknown。现有探针box/text=0.25，same-label NMS IoU>0.7 | GT匹配IoU阈值、>=或>、类过滤、可见框/全身框、边缘截断与无效框处理；不能借用NMS/文本阈值或几何容差 |
| 唯一性/同分 | 要求global one-to-one，不允许靠重复配对制造分母/命中 | 优化目标、贪心或二分图分配、同IoU/同置信度处理、稳定ID及输入重排不变性；目前没有已选Hungarian/greedy实现 |
| 指标与分母 | 名称为in_region_recall、in_region_recall_loss、boundary_retention、outside_suppression；目标指标当前均blocked且value/numerator/denominator=null。safe_rate零分母为null/no_support，不平滑 | 每项分子分母、基线与路由后配对、边界集合、按帧/图像/相机汇总权重；尤其recall_loss相对全部GT还是基线已命中GT、outside_suppression统计预测还是对象 |
| 拒判与负例 | 空pending objects不是负帧；unknown不是安全、正确排除或已审标签；失败/abstention保留。完整标注声明也仍返回evaluator_not_implemented | 未来合法已审负帧怎么贡献各指标、缺标/缺ROI是整单元阻断还是隔离报表、选择性风险/覆盖率和样本统计的精确分母 |
| 标签适用范围 | AI预标不等于人工GT；局部左右沿确认不扩成闭合ROI/门身份/事件GT。B第一版政策已确认，具体所选建筑/地图/门对应仍unknown | 独立标签来源、审阅范围、标签/ROI/相机/政策版本绑定、完整性与逐对象ignore定义；当前13个AI候选不能作为正式GT集合 |
| 反事实可辨识性 | v3空开发模板不准training/effect rows；A/B不强迫隐藏关系可见，C仅关键证据同等充分时检不变性；D同RGB的I0不可区分，I1-E端点配置执行不能算视觉识别增益 | 新正式来源及拆分、可观测性审阅标准、每个输入层的适用真值/拒判评估；不能从源几何真值反填图像证据充分性 |

## 最小待决清单（全部待审，不采用默认值）

- [ ] **对象与作用域**：先决定这四项是否只评逐帧person检测在ROI路由前后的保留行为；若是，门/关系/进出另立任务，不共享TP/FP/FN。明确global按原图/相机/类别如何分组、GT完整性与ignore范围。
- [ ] **候选边与阈值**：冻结匹配坐标/框类型、IoU计算、数值阈值与包含等号规则、类别/截断/无效框处理；标定只用合法开发/校准来源，禁止用待报测试集调规则。
- [ ] **唯一分配与同分**：选择并版本化目标与算法。可供审议的候选为“最大合法匹配数→最大总IoU→稳定GT/预测ID字典序”，也可以另选；这不是现行约定。必须决定重复框、完全同分与输入重排如何稳定处理。
- [ ] **四项计数账本**：逐项写分子、分母、基线、路由后对象映射与汇总权重；先匹配完整预测/GT还是分别匹配路由前后也须决定。把合法已审空负帧、无支持、缺标、缺ROI和unknown分开报告，避免删掉难例后报虚高分。
- [ ] **准入与反事实适用性**：冻结独立GT/ROI/版本、实际输入层、来源连通分量与留出隔离；单列I0不可辨识和I1-E配置执行，不混成视觉增益。现有16开发记录、13候选和局部人审均不自动获得正式评分资格。

决定以上内容只形成未来合同，不自动放开真实输入准入。还须合法原始包、对应完整性验收、合格独立标签及独立验证的评估实现；本轮不把缺这些材料解释为已确认缺69件。Mac六份快照/134个已核哈希也不能替代原包或GT。

## 可追溯依据

- [当前目标指标与safe_rate门禁](../continuous-cache-20261010/cache/vendor/roi_core.py)：target_metrics始终blocked，完成声明只改变阻断原因，不给正式分数。
- [当前私有验收入口与评分停点](README.md)、[人审适用范围](HUMAN_REVIEW.md)：文件验收/局部人审不等于scientific_ROI_admission或human_GT。
- [冻结反事实v3源码](../synthetic-counterfactual-method-20261011/protocol-v3/protocol.py)与[输入/标签分层草案](../synthetic-counterfactual-method-20261011/COUNTERFACTUAL_V2_ONE_PAGE.md)：历史空模板保留，不回写。
- [后续已确认B范围sidecar](../first-counting-scope-decision-20261012/README.md)：所选建筑封闭室内；不含露天院地/开放连廊；图像归属与S/B对应仍需证据。
