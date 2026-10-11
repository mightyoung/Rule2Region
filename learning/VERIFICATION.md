# v1 实际执行记录（软件核验摘要，非训练日志）

2026-10-11 UTC，云执行环境 `/workspace/Rule2Region-learning`；Python 3.12.14。独立分支 `task/boundary-learning-plan-v1-20261011` 基于公开 main `cf22ddd8e46ee83ddd9565b058e1df4e362377ea`。所有新增文件限定 learning/；没有更改 main、另一任务工作树、站点或旧 SOURCE_MANIFEST。

已读 README、发布说明/外部资产、现有 ROI 协议。开始时无项目简报/AGENTS.md/.agents/skills；后来远端 pilot 分支可读后，读取其 `850b4f3f3f9fc1ac955b16eaa070493a2e453b4d` 的 v1/v2 协议及真实子任务准入，未编辑或重跑其修复。

实际已跑：

- `python -m unittest discover -s learning/tests -v`：15 tests，12 通过、3 skipped（没有 PyTorch）；没有失败。
- `python -m learning.validate learning/data/real_manifest.json`：有效，真实样本 0。
- `python -m learning.validate learning/data/software_manifest.json`：有效，自制软件样本 3（每 split 一个独立身份；不是图像/楼宇数据）。
- train/evaluate 的 `--help` 可用；`python -m compileall -q learning` 成功。
- 实际尝试软件 train 命令，明确以 `PyTorch unavailable: training/gradient/profile NOT RUN; no dependency installation` 退出；未生成 checkpoint，不称梯度已跑。
- 发布文件清单通过 `python -m learning.manifest` 核对（清单排除自身与 Python 缓存）。根 SOURCE_MANIFEST 保留旧快照含义；learning 清单独立列新增文件哈希。

明确未跑：PyTorch shape/梯度、冻结动态断言、连续/恢复位级比较、真实编码器/模型推理、任何真实样本 AI 预标注、训练/LoRA/蒸馏、Kaggle 显存/耗时、完整拓扑 cut/固定计数器人数评估、完整历史测试。代码静态编译不能证明 PyTorch 运行正确。

没有下载模型/原视频/布局包、安装依赖、付费调用、读取/配置 Kaggle 凭证、上传原媒体/训练日志或站点发布。官方资料仅只读核查，来源见 acquisition_checklist 与 README。

下一步最小材料：合法同一空间的参考视图与允许场景说明；独立门/区域/同门核验及证据；building/layout-family/图纸/变体身份。证据不足保持 unknown。具备一个已安装 PyTorch 的环境后先跑软件测试，再安排真实基线；跨楼 train/calibration/test 需足够独立连接组件。完整人数验证还要所有边界/在途归属、独立事件/人数和初值盘点材料。父任务负责将本分支重要成果回存 Mac，本环境未执行 Mac 回存。

## e626 独立动态补充（父任务提供）

父任务在另一独立 detached `e626bc997207d453d58f47757de00e59398b57bb` 环境，使用现存 Python 3.12.14 / PyTorch 2.8.0+cpu / pytest 9.1.1，实际跑 `test_torch` 3 通过及全 learning 15 通过、0 skipped；无安装、权重或真实训练。验证覆盖梯度/shape/冻结/交换对称及恢复的模型参数等价；optimizer/RNG 完整状态等价、GPU 未验证。清单 22 文件核验，真实样本 0、软件样本 3。此补充不替换上面作者环境 12 通过/3 skipped 的历史，也不代表后续修复 head 动态通过。

## 增量复核修复（e626 后）

独立复核指出空 matches 候选导致 relation-only 校准整体失败，以及 known/unknown 最高分并列仍接纳 known。先加入标准库失败测试，再修改：空任务 threshold=null、task_statuses=not_applicable_no_candidates，评估 n=0/accepted=null/risk=null、曲线为空；有候选缺标签、没有可行覆盖、全部任务空、test 新出现却未校准的任务仍失败。unknown 优先于同分 known，所有非唯一最高 known 都拒判。

新增可选 PyTorch CLI 回归，使用本地软件头手动固定 logits，仅测试 relation-only calibration/test 的空任务状态，不构成模型结果。原源清单更新后包含新增回归文件。作者当前环境重新实际运行：20 tests，16 通过、4 skipped（仍无 PyTorch）；compileall 和两份 manifest 校验通过。新 head 动态执行及 GPU 尚未跑，交由父任务原独立环境复跑；未触碰该环境目录。
