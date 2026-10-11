# 增量修正 v2：准入分层，不读取 GT 决定准入

当前没有识门或室内外判定模型。物理门、端点区域、membership、ROI、方向均预置。模型两个槽位与真实视觉仍 NOT_RUN；此次不训练、下载或调用模型。

打开 [修正交互报告](report_v2.html)，对照 [原报告](../building_boundary/report.html) 与 [旧版三个完整反例](v1_counterexamples.json)。原 v1 文件及 manifest 全部未改动，其有限夹具零误差保留。

## 修复结果

| 旧版假准入 | 旧整栋 max / 事件 FP / FN | v2 修复 |
|---|---|---|
| 只反转 line 端点，a_side 不变 | 2 / 3 / 3 | 根据候选有向线与独立 a/b 区域侧锚点拒绝 |
| 合法但错位 ROI=[.01,.01,.1,.1] | 1 / 0 / 3 | 数值合法仍不能通过线 / ROI 一致性及审核通道覆盖检查 |
| 初值改 30/20/10，旧配置审批仍有效 | 27 / 0 / 0 | 配置审批层仍可有效；旧初值会话资格失效，无运行 / 分数 |

v2 先分数值合法、有限线/ROI 内部有正长度交集、经审核场景通道/区域侧证据三层。只合法坐标或线/ROI 相交不证明门位置正确。支持区域和 a/b 锚点是独立自制标定参考，不是 GT 事件或人数；不精确匹配原候选线。同步反转线和 a_side，以及改用 x=.48 的合规线，经新审批均通过且原有限夹具误差 0。

配置审批绑定准入版本、config、topology、目标合同、场景证据及参考资料的规范 JSON SHA-256。初值会话资格另绑定整个 session（含 t0/时间基准、初值、来源 SHA / 版本、session_id/revision）、配置审批 SHA 和准入版本。初值、t0、来源、版本或审批变更使旧会话资格失效；配置改变也需重新绑定会话资格。来源数值一致是来源完整性检查，不是拿 GT=3 作硬编码门禁。

所有审批/盘点/场景证据是作者模拟，只能用于合成小试，不是人员身份认证或真实现场核验。重新提供并确认自洽的 30/20/10 盘点来源会被允许；软件不能识别伪造来源，真实正确性须外部审核。这一正对照特意证明没有以原 GT 数值伪装自动判断。当前固定执行器只支持 synthetic_step t0=0，不宣称支持任意真实时间会话。

`pilot_v2.execute` 是唯一修正准入入口；配置与会话任何一层失败都没有预测/正式成绩。准入通过后复用原 `v1.execute(..., diagnostic=True)` 的数值计数路径，仅消费 series/events，显式绕过已知有缺陷的旧准入层。过线、累加、观测和评分未换实现；没有将旧 diagnostic 身份当新审批结果。真实应用必须重新设计认证审批，而非调用这些 fixture_* helper。

## 复现

```sh
python pilots/building_boundary_v2/reproduce_v1.py
python pilots/building_boundary_v2/pilot_v2.py
python -m unittest discover -s pilots/building_boundary_v2 -p 'test_*.py' -v
python -m unittest discover -s pilots/building_boundary -p 'test_*.py' -q
python pilots/building_boundary_v2/build_report_v2.py
python pilots/building_boundary_v2/manifest_v2.py --check
```

18 项新增测试、原 20 项测试通过；12 项增量拒绝情形与 2 项不同合规几何正对照已运行。`browser_check_v2.py` 在已安装 Playwright/Chromium 环境做 10 项界面检查，未安装环境不自动下载。新报告生成器读取旧模板和数据，写到新目录。`fixtures.py` 只编制合成坐标参考与独立初值盘点，不读取评分 GT。

准入模型：`geometry_evidence.json` → `scene_reference.json`；`initial_session.json` → `initial_census.json`。规范 JSON 摘要用于收据依赖，文件字节 SHA 单独由 MANIFEST.json 记录。两个摘要不是相同算法序列化。

[修正与下一阶段协议](PROTOCOL_V2.md) 已把视觉 door 候选、门连接区域、室外证据、ROI/方向审核和固定视觉计数拆开。最小真实资料须来自同一空间，包括各门参考帧/片段、完整通道/楼层拓扑、确认室外端的图纸或现场记录、标定侧证据、t0 初值盘点来源和独立人工流量/占用 GT。视觉门框不能替代区域边界证据。本轮只冻结协议。
