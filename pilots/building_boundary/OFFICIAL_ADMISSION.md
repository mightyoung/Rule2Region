# 真实公开子任务准入 — 官方资料核查，2026-10-11

本阶段只读官方介绍，未下载原媒体、GT 大包或布局包，未运行真实视觉。没有核实到同栋视频 + 全门/层拓扑 + 完整人流/占用 GT 的闭环组合。此结论是本次核查范围的缺口，不是对所有公开数据的不存在证明。真实账本为 `real_gt_ledger.template.json`，各策略真实成绩均 null / NOT_RUN。

| 官方源 | 可用的未来子任务 | 缺失/不适用 | 本阶段准入 |
|---|---|---|---|
| [CAVIAR](https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1/) | 商场店门的进入/离开及同门两个视角的责任候选；如 EnterExitCrossingPaths2、OneLeaveShopReenter1、TwoEnterShop1，后续只选最小片段 | 人工框/活动 XML 不自动提供完整店门/整楼边界、初值和全楼占用 GT；店铺门不能标作整楼外门。官方提及某些不动的人未框注 | DOCUMENT_VERIFIED；真实评估 NOT_ADMITTED |
| [ChokePoint](https://arma.sourceforge.net/chokepoint/) | 单 portal 三镜头责任选择及 entry/leave 方向标识子任务 | 主任务是脸身份验证；手工眼点用于脸裁剪，不是全身过门/全楼占用 GT。两个 portal 采集时间相隔一个月，不能拼同栋闭环 | DOCUMENT_VERIFIED；真实评估 NOT_ADMITTED |
| [ProcTHOR-10K](https://github.com/allenai/procthor-10k) | 布局/门连接元数据合同，未来独立检查图结构配置 | 住宅布局数据，不是现成多层人流仿真、真实视频或完整人数 GT；不能把生成布局或本夹具称真实实验 | DOCUMENT_VERIFIED；布局包未获取，真实视觉不适用 |

## 使用条件与获取闸门

CAVIAR 官方页标示 CC BY-SA 与项目署名要求；下载前核实选定文件和具体许可版本。本次仅记录出处，不再分发原媒体或标注。

ChokePoint 官方条款限非商业研究，要求 NICTA/论文署名、保留许可和修改标识。官方 GT 说明是手工眼点，原数据规模约 12 GB；不下载包，不把身份验证协议改称人数协议。

ProcTHOR 官方仓库标示 Apache-2.0，并通过 prior 加载住宅数据；此次没有安装 prior 或拉取数据。使用布局时需核实版本、具体文件 SHA 与依赖/资产条款。

每个未来子任务须单独提交：明确空间/物理门与双向定义、责任镜头清单、选段/参考帧本地 SHA、人工穿越事件 GT（含歧义和独立复核）、候选配置 SHA/人工确认、固定视觉执行器版本。占用任务另需初值、所有边界通道覆盖证据和在途归属。缺任一项保持 UNKNOWN。禁止拼异构数据称真实闭环。

同步与画面可见作为当前假设，不做研究变量。CAVIAR 官方记录了片段起点对应和重复帧现象；今后获取时只作为输入适用边界核验，不扩展成本阶段时间错位研究。
