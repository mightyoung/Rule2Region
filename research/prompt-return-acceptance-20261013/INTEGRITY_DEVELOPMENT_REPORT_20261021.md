# 原包验收状态完整性：2026-10-21开发增量

研究基线与父提交：`2125cbb09b36ae3737412d208612df2aa36a6ae0`。
独立分支：`task/acceptance-integrity-20261021`，保留该任务分支全部依赖。
交付完整提交SHA由Git提交记录和交付消息提供，不在同一提交中伪造自引用SHA。

## 原任务与冲突核查

保存云环境原checkout是`cf22ddd8e46ee83ddd9565b058e1df4e362377ea`源码快照，
未将其当研究基线。已读取任务头最新提交、验收README、原五项待决清单、
反事实独验交接、设计状态、研究交接任务和最新公开进度；仓库及上层可读目录
未发现AGENTS.md，也未发现另一个简报文件，故沿用这些已发布入口。

`git ls-remote --heads origin`与fetch核实原任务头仍为上述2125cbb，
CPU/GPU/脱敏回执三分支仍为README中固定SHA。GitHub只读查询
`repo:mightyoung/Rule2Region is:open`返回无开放issue，开放PR仅历史交接PR #1。
未发现相同增量的活动公开任务；不可据此声称不存在未公开任务。
工作区初始干净，原work分支保留，未覆盖用户修改。

## 实质变化与测试先行

已有69件清单、失败回传、人审实际目录门禁不重复实现。本次仅修复RUN状态边界：

- `schema_version=true/1.0`不再等于整数版本1；trial退出码必须为真正整数。
- completed trial必须显式退出0；running尚无退出码、failed现场负整数退出码仍保留。
- 非对象状态及非法trial身份/状态形状抛出ValueError，进入既有CLI失败记录路径。
- RUN_PRIVATE.json重复字段（含嵌套trial）、NaN/Infinity字面量和溢出浮点数拒绝。
- synthetic CLI试验确认仅写PRIVATE_FAILURE，不生成完整性通过报告或人审模板，
  原输入状态字节不改。

原验收43项先通过；新增测试在实现前得到17个失败、6个错误（含subtest）。
数值溢出追加用例也先得到2个失败，再补有限浮点解析。最终9项新测试全部通过，
验收目录合计52项通过。所有新增材料为明确标注SYNTHETIC的状态或极小文本变体，
无真实媒体、真实张量、真实模型或真实包。合法状态单测通过只表示状态合同通过。

[执行回执](INTEGRITY_DEVELOPMENT_VALIDATION_20261021.json)保存实际argv、退出码、
输出尾部和完整输出SHA256。末轮另从Git加载基线checker到独立Python进程进行
回归敏感性检查：9项测试出现19失败、6错误，退出1；其中CLI子测试仍执行当前
checker脚本，只有直接加载的checker模块回退到基线。该检查不是历史实验复现。
最初RED日志与末轮日志仅存本环境`/tmp/rule2region-acceptance-20261021`，不含私有数据。

## 实际命令与终态

| 命令 | 实际终态 |
|---|---|
| `python3 -m unittest discover -s research/prompt-return-acceptance-20261013 -p 'test_*.py' -v` | 52项通过，退出0 |
| `python3 research/prompt-return-acceptance-20261013/audit_returns.py --repo . --out /tmp/rule2region-acceptance-20261021/public-audit-final.json` | 19份固定公开回执复核通过，退出0；private_original_package_verified=false |
| `python3 -m pytest` | 退出1，No module named pytest；全仓测试未启动，未安装依赖 |
| `git diff --check` | 退出0 |
| 基线checker回归敏感性检查 | 实际完整argv见回执；退出1，预期RED，19失败/6错误 |

独立内部只读复审实际重跑52项验收测试与diff检查均退出0；另用synthetic
`1e999/-1e999/NaN/Infinity/-Infinity`探针确认均抛ValueError。复审发现的两项反馈
已闭合：为溢出浮点补测试与解析检查；纠正“完全重复预测只能一个获配对”的建议
表述，明确不同重复预测仍可各配不同重叠GT。复审结论：当前增量无阻断问题。
这不是对原图、标签真实性或完整私有运行的独验。

## 五项待决与保持的停点

[最小决策包](SCORING_DECISION_RECOMMENDATIONS_20261021.md)逐项提出对象/作用域、
候选边/阈值、唯一分配/同分、四项计数账本、准入/反事实适用性建议及影响。
每项均为proposed、approval=null。未实现正式评分器，未把建议写成用户批准。

严格JSON解析仅适用RUN_PRIVATE，不声称全仓JSON解析都已加固。
原包完整成功路径仍未执行，不能从局部状态验证推定成功；全仓pytest因环境缺依赖
未启动。冻结v2/v3、指标实现、模型/输入manifest、原回执和历史结果均未修改；
本次未重跑冻结v2/v3套件。没有新forward、训练、付费、真实模型连接、私有数据连接、
凭据配置、权限扩大、部署或合入main。

原始完整私有回传包、完整人工GT、闭合ROI、身份与S/B具体对应继续缺少可验材料。
B所选建筑封闭室内政策已确认，但具体对应unknown；准确率null。
`human_GT=false`、`scientific_ROI_admission=false`、`decision=unknown`。
下一步只需有权决定者对五项分别记录采用/具体修改/暂缓；未答复继续待决，
即使全部批准也仍须补齐真实材料与独立评分实现验证。
