# 整栋 / 楼层边界配置小试

独立候选应用、小规模合成软件闭环。先读 [冻结协议](PROTOCOL.md)、[基线核查](BASELINE_CHECK.md)、[真实公开子任务准入](OFFICIAL_ADMISSION.md)。打开 [report.html](report.html) 可交互查看拓扑、逐人状态、责任镜头、方向与错误计数。报告没有联网依赖。

```sh
python pilots/building_boundary/pilot.py
python -m unittest discover -s pilots/building_boundary -p 'test_*.py' -v
python pilots/building_boundary/build_report.py
python pilots/building_boundary/manifest.py --check
```

全部使用 Python 标准库。`make_fixture.py` 是固定自制夹具的可复现作者脚本，不生成视频；文件已冻结，评估不重新生成或调参。报告和结果含合成数据，禁止引用为真实或模型成绩。

## 结果与可得结论

manual_truth、metadata_rule 在全部 12 时间点/6 目标误差为 0。三外门、同门多视角、内部换房、楼梯、电梯、初始非零和进后出均覆盖。metadata_rule 是透明确定性编译器：给定完整且已解释的区域/通道与各镜头几何、方向，计算业务目标的边界并指定首个责任镜头。它不从画面识别门，不理解任意自然语言，不自动估计 ROI，不是 LLM；ROI/线段来自作者定义的元数据。模型两个槽位 NOT_RUN，成绩 null。

仅 door 候选用作者门清单的全门作为每个占用目标边界，遭拒绝。它不是实际 door 检测器调用；几何与唯一责任镜头给予同样的理想元数据，以单独检查边界 membership 的影响。强制 diagnostic replay 显示错误配置的偏差，不是正式策略成绩：整栋 MAE=0.917、最大误差=2，1 层 MAE=1.333、最大误差=3；整栋末值偏差恰好为 0，说明只看末值会掩盖中途错误。漏服务门、漏楼梯/电梯、双 owner、反方向、初值缺失、ROI 缺失、无审批、审批后改变配置、完整覆盖证据未知等九种失败闸门已实际测试。不存在真实审核耗时或修改数证据，不能声称降低配置成本。

固定执行器是坐标线段过线计数器，所有策略使用同一输入 SHA/同一实现；不是 YOLO 或真实视觉跟踪器。GT 由独立手工人物位置快照给占用，由手工物理事件给方向/人次；不会读计数器预测。事件多重集合按时间步/物理门/方向评分，GT 身份不传给执行器。同门备份镜头不累加；重传按局部 track_id/step 去重。

## 合同与对照接入

`topology.json`：区域→楼层；门→a/b 端点→可用镜头→各画面 line/roi/a_side。
`goals.json`：明确 membership 的 occupancy/external_net 目标与独立初值。
`manual_config.json`：人工真值配置示例。模型候选也须使用同一结构，不得新增统计器。

接入自己的候选时，调用 `validate(config, topology, goals)` 与 `execute(config, topology, goals, observations, review)`，再用独立 GT 调用 `score(run, truth)`。小试 `fixture_review` 仅模拟夹具审批；真实确认须另外提供审核者、时间、配置 SHA 和人工修改记录。本模块不支持生产审批，不能拿模拟收据上线。没有审批或配置 SHA 变化即拒绝。

评估中只做有限例子的合同证明；完整通道清单由作者/未来现场审核保证，不能识别未列入拓扑的隐蔽门。计数覆盖不等于物理覆盖已被视觉验证。楼梯/电梯原子转移省略在途区域，后续真实协议须处理其归属。当前假设同步且完整可见，不研究错位或不清晰问题。

真实公开子任务只完成官方资料准入，未下载数据且无真实分数；真实 GT 空模板不等于零人。下一步最小输入见协议末节：同一空间小片段与完整拓扑、参考帧/合法来源、本地哈希、独立初值及人工事件/占用标注。之后先比较模型候选配置与人工配置，固定视觉执行器；本阶段没有训练、视觉调用、账号密钥或付费。

## 发布范围

仅本目录自制 JSON、Python、Markdown、HTML 与 manifest 推独立分支；原媒体不上传。原 SOURCE_MANIFEST.json 是旧公开快照清单，本小试用自己的 MANIFEST.json，不改旧清单、README 或旧协议。远端提交与 manifest 核对另见交付记录。
