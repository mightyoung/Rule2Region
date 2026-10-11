# 站点待标注合同 v1（交接；没有修改或发布站点）

结构 schema 为 `data/manifest.schema.json`；还必须跑 `python -m learning.validate` 做跨样本语义检查。`data/real_manifest.json` 当前为空；`software_*` 只为自制张量的软件测试，不得出现在真实样本页面或模型结果报告。待获取公开来源见 `data/acquisition_checklist.json`，其中候选序列名只是获取清单，不是已经得到的样本。

## 最小材料清单

- 合法来源/许可、访问日期、用途许可核验；本地视图 SHA-256、相机/视图编号和可引用证据。Git/公开站点清单只放元数据，原媒体留在授权本地存储。
- building_id、layout_family_id、variant_family_id；物理门集合只作分组元数据。源图纸 SHA（没有图纸则 null，存在但身份不明就暂停分割），变体/多视角/连续片段必须共享来源身份。ID 应全局带来源命名空间。
- `input` 只放可允许模型看见的视图引用、场景说明、候选区域、门框与候选关系，不放人工物理同门映射、核验结论/理由、完整 GT、独立人数或审核证据。允许的图纸若本就揭示拓扑，所有基线都获得同样信息，单独记录这个条件。
- 两侧真实区域、同门关系和“是否足以判断”由独立证据决定。完整性核验还需现场或完整图纸确认隐藏通道、楼梯、电梯、连廊、室外、目标房间/楼层/整栋集合；此完整图保存在独立 GT 包，不塞入关系模型的特征缓存。

## 标注规则

`region_pairs` 每项用局部 ID 指向门与两个候选区域，标 `connected / not_connected / unknown`。connected 指该门/通道直接连接这两个区域；not_connected 需支持排除的证据，不能把未看见当负例。`view_pairs` 为两个视图门实例，标 `same / different / unknown`；人长得相同不是同门证明。两个关系都无方向；投影的进出方向属于后续固定计数器合同。

`sufficiency` 对每个区域对标 `sufficient / insufficient / unknown`：能判断 connected 或 not_connected 都可以 sufficient；明确缺乏两侧身份为 insufficient；证据充分性本身也无法核验为 unknown。GT unknown 是可监督的第三类，不是缺失标签；未标注用 `truth: null`，不得用 unknown 把缺材料伪装成已复核标签。AI 对证据的猜测不能生成独立真值。

每个 `input` 候选在审核后的三组标签 map 中都要有对应标签。门候选坐标为归一化 xyxy，区域局部 ID 只用于引用，网络输入只有特征；跨场景固定 ID 分类不在合同内。当前代码不产生门检测候选，不生成拓扑完整性证书。

## AI→用户复核→历史

AI 预测写 `ai_preannotation`（模型精确 revision、labels、risk、evidence/labels/rules 三版本）；状态是 `proposed` 或 `default_adopted_pending_review`。普通预标注可默认采用作为候选配置，但 `review.status` 仍 pending、truth 仍 null。高风险、unknown、缺预标注置顶。不会因每次刷新或置信波动重复打乱已复核历史。

用户确认/修正后才写 `truth`，review 为 confirmed/corrected，附 reviewer、证据引用、匹配的三版本。不确定也可经人工核验确认 unknown，但需说明究竟缺少什么证据。站点作者把完整旧记录另存追加式历史，以 `id + versions` 去重；本 manifest 只保存最新有效记录。

仅 evidence/labels/rules 版本变化才使已复核项进入重审；`learning.review.review_key` 和 `review_queue` 是纯交接辅助函数，可识别历史/重审与优先级，不写站点。重审时先归档旧 truth/review，把最新状态复位 pending、truth=null、review versions=null，再生成匹配新版本 AI 预标注。不能复制旧审核到新版本；validator 拒绝这种 stale 真值。

## 分割与独立真值

分割前建立 building/layout-family 联合连接组件：共享楼宇、布局族、物理门、源图纸哈希、变体族或相同媒体的任一记录都只能在同一 split。若这些连接传递地把记录连起来，整个组件一起分；缺身份则不做正式评估。用固定种子分配组件到 train/calibration/test 并保存 manifest，再派生 augmentations。`validate_manifest` 检查所有共享身份和相同媒体哈希不得跨 split；它不能发现错误命名或视觉近重复，仍需来源审计。

独立 GT 应由现场/原始合法图纸/另一复核者的证据构建；不要只确认 AI 的解释而没有证据。test 标注与 input 特征生产分离，缓存生产者不得读 truth/review/identity。当前 `input_only` 声明和 input 哈希仅提供追溯，不证明编码器实际没有偷读 GT，后续必须审查生产代码和记录。规范及 schema 不代表实际人工标注已经完成。
