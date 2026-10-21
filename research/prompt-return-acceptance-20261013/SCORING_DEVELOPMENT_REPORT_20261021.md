# 已批准五项规则的评分器：开发交付与独立复审

基线及父提交：`d87bdb2221187bcf4ba5405a8b1501898cbae561`；
分支：`task/acceptance-integrity-20261021`。交付完整提交SHA以Git提交记录和最终交付消息
为准，避免同提交自引用。远端工作开始时仍为该基线，工作区干净；没有新AGENTS.md。
已重读最新交接/报告及五项建议，未发现其它远端分支前移；没有覆盖用户修改。

## 批准来源与完成范围

[批准记录](SCORING_APPROVAL_20261021.json)保存本轮受信委派中的用户原话、上下文、
被批准建议的精确commit/path/SHA256，以及五项逐项状态。用户消息精确时间未给出，
记null；记录日期不冒充消息时间。此前建议与待决清单保留原文，最新入口指向批准记录。

[契约v1](SCORING_CONTRACT_V1.md)与`scoring_contract.py`落实五项：

1. 逐帧person/可见框与完整标签绑定；普通GT优先匹配，ignore只处理余下预测。
2. 连续边坐标、精确十进制有理数、裁剪审计和IoU>=1/2；非法输入阻断整单元。
3. 最大基数→最大总精确IoU→稳定ID字典序；只匹配完整基线一次，路由后不重配。
4. 四项批准分母、合法负帧、零支持null、unknown整单元/总体阻断、同臂micro及相机支持。
5. 来源连通分量跨split阻断、固定总体、输入层/反事实适用范围与真实准入关闭。

纯stdlib参考实现，未将浮点近似用于匹配或几何判断；支持简单凹ROI，孔洞/多部件
表示不在当前接口范围，遇到这类输入须阻断并先扩展合同验证，不能偷偷用外接矩形。
I1-E独立臂只有配置执行解释，不混入视觉增益；同臂禁止混输入层或split。
跨臂固定相同来源集合和实际完整GT/ignore/ROI/尺寸，版本字符串不足以证明内容一致。

当前可运行入口仅为[synthetic软件fixture](SCORING_SYNTHETIC_FIXTURE.json)。
fixture_admission只是测试门禁标记，不是来源或人工审阅认证；所有输出的
human_GT=false、scientific_ROI_admission=false、accuracy_metrics=null、decision=unknown。
真实输入不能用这些标记自行放行，返回real_source_admission_not_implemented。

## 测试先行、实际终态与失败

完整argv、环境覆盖、输出尾部及完整输出SHA见
[执行回执](SCORING_DEVELOPMENT_VALIDATION_20261021.json)。日志仅保存在本环境
`/tmp/rule2region-scoring-20261021`，均为软件/公开对象检查，无真实私有数据。

| 实际命令 | 最终终态 |
|---|---|
| `python3 -m unittest discover -s research/prompt-return-acceptance-20261013 -p 'test_scoring*.py' -v` | 27项通过，退出0 |
| `python3 -m unittest discover -s research/prompt-return-acceptance-20261013 -p 'test_*.py' -v` | 79项通过，退出0（包括原52项） |
| `PYTHONPATH=/tmp/rule2region-scoring-20261021/testdeps python3 -m pytest -o addopts= -q`，后接README推荐的8个测试文件（完整argv见回执） | 52项通过，退出0 |
| `python3 research/prompt-return-acceptance-20261013/audit_returns.py --repo . --out /tmp/rule2region-scoring-20261021/public-audit.json` | 固定19份回执复核，退出0；原私有包未验 |
| `python3 research/prompt-return-acceptance-20261013/scoring_contract.py --input research/prompt-return-acceptance-20261013/SCORING_SYNTHETIC_FIXTURE.json --out /tmp/rule2region-scoring-20261021/fixture-final-result.json` | synthetic_scored，退出0；不是研究效果 |
| `git diff --check` | 退出0 |

实现前15项核心测试实际RED（退出1、15失败），独立匹配oracle最初6项也实际RED
（退出1、6失败）。首次实现21项有1项CLI失败：本环境`/tmp/.git`是挂载标记，
仅检查存在会误判实际checkout，已改为沿用git rev-parse核实真实checkout再拒绝输出。
复审反馈新增测试先得到18项中1失败2错误，再修复；共享参考集合冲突新增4个
subtest先失败，再加入内容签名核验。最终无未闭合的评分测试失败。

全仓`python3 -m pytest`初次因pytest未装退出1。仅在/tmp隔离安装pytest及其小型依赖
（安装退出0，不改项目依赖），随后全仓命令在收集阶段退出2，共9错误，未执行完整套件：

- `tests/test_audit_virat_videos00_media.py`：缺cv2。
- `tests/test_build_codec_derivatives.py`：缺av。
- `tests/test_build_development_video_receipt.py`：发布快照无research_tools.remote。
- `tests/test_codec_fixture.py`：缺av。
- `tests/test_evidencegate_smoke_inputs.py`：缺av。
- `tests/test_g2_smoke_contract.py`：导入链缺av。
- `tests/test_g3_formal_contract.py`：导入链缺av。
- `tests/test_g3_latency_contract.py`：导入链缺av。
- `tests/test_g4_build.py`：导入链缺av。

没有用删除这些测试或伪造未发布模块来报告全仓通过；不为本任务下载模型/媒体。
52项轻量测试通过只覆盖其明确范围，不替代全仓结果。

## 独立复审

两位内部复审者实际运行评分套件27项均通过、diff检查退出0；匹配oracle复审者还
独立复跑验收目录79项通过。没有将派发记作执行。独立测试采用不同实现：
300个固定seed小图穷举三目标最优解、30次完全同分重排、精确有理数微差，
以及205个凹U形ROI框用不重叠矩形面积分解oracle对照（含断开交集和反向顶点）。

复审发现并已闭合：全局来源阻断仍显示全准入；同臂混输入层/split；跨臂不同总体；
同来源/版本但实际GT/ROI/尺寸不同。每项均补失败测试后修复。最后独立重放绕过探针
正确blocked、admitted=0、总体相机支持为空。最终结论：当前synthetic软件合同无阻断问题。
这不是对真实标签、完整原包、来源真实性或模型效果的独验。

## 仍未执行与剩余阻断

真实原包验收、完整人工GT/闭合ROI/具体身份与S-B对应验证以及真实评分均未执行：
所需材料仍未提供，本次规则批准不补足这些事实。B所选建筑封闭室内政策保持确认，
具体对应仍unknown，准确率null。真实评分目标处于缺证据阻断，非权限拒绝或待重试动作。
冻结v2/v3、旧target_metrics、原始结果和模型/输入manifest未修改，本次未重跑冻结套件。
没有新模型forward、训练、付费、真实私有数据/模型连接、凭据配置、权限扩大、部署或合main。
