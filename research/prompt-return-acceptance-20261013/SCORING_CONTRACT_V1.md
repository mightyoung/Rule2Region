# Person ROI retention v1：已批准规则的确定性参考评分器

版本`person-roi-retention.v1`，基线`d87bdb2221187bcf4ba5405a8b1501898cbae561`。
[批准记录](SCORING_APPROVAL_20261021.json)保存用户原话、批准对象的提交/路径和范围；
来源消息精确时间未提供，记null，记录日期不冒充批准时间。
原[五项建议](SCORING_DECISION_RECOMMENDATIONS_20261021.md)是历史待审快照，不回写。

## 执行边界与接口

`scoring_contract.py`是独立的stdlib参考实现。它不接入旧`target_metrics`，不更改
冻结v2/v3。当前只接收`kind=synthetic_software_fixture`；所有输出均有
`human_GT=false, scientific_ROI_admission=false, accuracy_metrics=null, decision=unknown`。
数值仅放在`software_metrics_by_arm`和synthetic单元账本，不是本研究准确率。
任何真实输入，即使自行填写全部准入声明，也被`real_source_admission_not_implemented`
阻断。真实入口须日后以实际原包和完整独立人工GT另行验证，不靠开关放行。

JSON入口`loads(text)`拒绝重复字段、NaN/Infinity。小数解析为Decimal，再转Fraction；
坐标按原始JSON十进制数值精确计算。Python接口只接收int/Decimal坐标，拒绝bool/float，
避免原JSON精度已经丢失；不要先用默认json.loads转float。IoU、面积和匹配权重无epsilon。
结果中的原框/裁剪框、匹配IoU以规范有理数字符串输出，指标保留整数分子分母，
非零率的value是其常规浮点显示值；任何判断均不用显示浮点值。

顶层只允许`contract_version, kind, preregistered_units, units`。预登记ID清单必须非空、
唯一并与输入单元精确一致。增删单元或重复ID阻断总体。空清单不获得“完美评分”。
最小完整输入见[明确标注synthetic的fixture](SCORING_SYNTHETIC_FIXTURE.json)，
不得将其来源hash、准入标记或几何冒充实际材料。

## 1. 对象、绑定及ignore

每单元只评`task=person_roi_retention`、`category=person`。包含unit_id、原图SHA256、
camera_id、source_version、arm、policy_version、width/height、split与lineage_ids。
同原图/相机/版本/arm只允许一个单元；不同提示臂独立匹配并分别micro汇总，不跨臂相加。
不同来源/相机/时间从不共用匹配图。门/结构/建筑/关系/事件不进入TP/FN。

labels必须为complete、annotation_complete=true、scope=full_frame_visible_person、
fixture_reviewed=true。binding精确绑定原图hash、相机、来源、ROI版本、政策版本。
所有普通GT与ignore拥有唯一稳定ID（两集合不能重叠），预测在自身集合内ID唯一。
稳定ID在结果之前冻结；重排列表不能重建ID。所有框保留，不悄悄去重。
每个ignore必须有预先确定的非空reason。先匹配普通GT；再把未匹配预测与ignore按
IoU>=1/2比较，命中仅入ignore库存、不贡献outside分母、不抢普通GT配对。
此参考器不生成完整TP/FP精度报表。

## 2. 几何、候选边及非法输入

GT与预测先与图像矩形裁剪，连续边坐标面积无`+1`。原框须宽高为正且有限，裁剪后
面积仍须正；否则整单元阻断。GT/ignore越界裁剪须truncated=true，所有对象均须显式
truncated布尔审阅标记。预测越界合法时保留原始/裁剪框两份审计字段。
普通同类别边仅在精确IoU>=1/2时存在，等号纳入，触边不匹配。

roi有version和points，要求显式首尾闭合、至少三不同顶点、面积正、全部在图像内、
无自交/重边/回折重叠。支持简单凹多边形；本版不接收孔洞或多部件ROI表示。
多部件/孔洞输入不能静默取外接矩形，须后续新增合同表示及验证后才能接收。
分类使用裁剪后的框和已审ROI：正面积交且仍有面积在外为boundary，完全在内为in；
无正面积交但接触为touch；既不交也不接触为out。boundary属于G_in。
GT/预测touch均单列，不纳入in/out分母。ROI和框分类与route无关。

未知字段、缺尺寸、身份/版本unknown、无效框、非person类别、重复ID、未审ROI/
标签/忽略原因、未知route、binding不符均整单元blocked；不按FP吞掉坏输入。

## 3. 唯一分配与冻结基线

`maximum_matching(edges)`输入`{(gt_id,prediction_id): Fraction(IoU)}`。
优化顺序严格为最大配对数、最大总精确IoU、排序配对序列字典序最小。
参考实现使用最小费用最大流，费用是有理数与整数的有序二元组；增广到最大流后
第一费用最小即总IoU最大，第二费用用按ID排序边的二进制优先位确保最后的字典序。
不以单一浮点加权和混合三目标。独立穷举oracle按三个目标直接枚举验证。

所有完整预测与普通GT只匹配一次。配对M冻结后仅查询所配预测的keep/suppress，
不在路由后重匹配。完全重复框保留不同ID，最多每GT/每预测各一个配对；
多个重复预测可分别配不同的重叠GT，剩余框是未匹配库存。

## 4. 四项账本、汇总及阻断

记G_in为in/boundary普通GT，G_bnd为boundary子集，M为完整基线配对。

| 指标 | 分子 | 分母 |
|---|---|---|
| in_region_recall | G_in中M配对预测keep数 | 全部G_in |
| in_region_recall_loss | G_in中M配对预测suppress数 | 全部G_in |
| boundary_retention | G_bnd中M配对预测keep数 | G_bnd中基线已配对数 |
| outside_suppression | 非ignore的out预测中suppress数 | 全部非ignore的out预测 |

outside是预测库存率，包含未匹配框，不解释为正确排除人体。报告边界GT总量、基线
漏检、重复预测框、未匹配预测、ignore与touch排除库存。合法完整空负帧前三项0/0，
value=null/reason=no_support；仍可对outside提供支持。pending空集合阻断。

按arm合并整数分子分母做micro；提供同臂相机分层支持量。不将无支持填0，不平滑。
同arm不得混input_layer或split；各arm须有相同的原图/相机/来源/政策/ROI版本集合，
否则总体阻断。不同输入层的独立arm仍可各报软件账本，但cross_arm_comparison明确
not_comparable_across_input_layers，I1-E范围始终只属配置执行，不解释成视觉增益。
同绑定键还须核对实际完整GT/ignore内容、ROI几何与尺寸，版本字符串相同不能掩盖
不同参考集合。比较使用精确坐标；对象列表重排和多边形循环起点/方向变化不算改动。
任何预登记单元blocked，整个总体和每臂主指标都blocked（value/numerator/denominator
全null）。已通过单元可保留明确synthetic的单元账本用于诊断，不冒充总体；
coverage、blocked_reason_counts、unknown_routes明示未决和缺失，不能删困难单元再报总体。
coverage另列locally_valid_units；全局来源/聚合规则失败时admitted_units归零，不能
显示“全准入”。camera_support_by_arm只在总体可计算时填数；已局部校验的子集库存
独立命名为locally_valid_subset_camera_support_by_arm，保留诊断而不充当总体支持。

## 5. 准入、来源连通分量与输入层

fixture_admission是synthetic软件测试标记，必须恰有九项true：source_authorized、
bundle_integrity、independent_full_frame_labels、roi_reviewed、camera_registered、
policy_bound、observability_reviewed、stable_ids_frozen、preregistered_before_results。
它们模拟门禁条件，**不认证任何真实来源或人工审阅**。

lineage_ids表示同视频近邻、同场景派生、合成族等来源连接；同原图SHA也自动相连。
实现求传递闭包，任何连通分量跨development/calibration/holdout则总体阻断。
缺lineage或未知split阻断，声明本身不是独立来源证明；真实输入仍不可执行。

input_layer限定I0、I1-R、I1-R-no-RGB、I1-E。前三者不能target_answers_supplied=true；
I1-E只解释configuration_execution_only，绝不解释为视觉增益。隐藏关系输出unknown，
任何层均禁止source_geometry_substitutes_observability=true。
counterfactual_family为none/A/B/C/D；C还须equivalent_evidence_reviewed=true。
A/B不强迫隐藏关系可见，D的同RGB关系不因框匹配而变可辨识。
B已确认的所选建筑封闭室内政策保持；具体建筑/门/地图/S-B对应仍unknown。

## CLI与终态

```sh
python3 research/prompt-return-acceptance-20261013/scoring_contract.py \
  --input research/prompt-return-acceptance-20261013/SCORING_SYNTHETIC_FIXTURE.json \
  --out /tmp/rule2region-synthetic-scoring-new.json
```

输出必须在实际Git checkout外，独占新建，不覆盖旧记录。退出0仅synthetic_scored；
退出2为blocked（含真实输入）；参数/旧输出/IO错误非零且不得覆盖。
没有真实原包/GT时不能拿此命令报告研究效果。
