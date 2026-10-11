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
