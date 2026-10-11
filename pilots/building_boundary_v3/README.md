# v3 有界补修：I1 有限线 / M1 来源字段

[增量交互报告](report_v3.html)保留旧反例与新拒绝结果，展示真实有限横切条件。[协议](PROTOCOL_V3.md)记录最新 C0/I1/M1 分级与适用限制。v1/v2 的报告、文件、manifest 原样保留。

先写回归测试并对 v2 运行：4 tests，11 个失败（两个漏洞及来源空值/类型变体），1 个不同完整线正例通过；日志 `RED_PHASE.txt`。随后 v3 的 9 项作者测试通过，另复跑旧 38 项软件测试。有限线横切检查不读事件、人物轨迹或占用 GT，模型与真实视觉保持 NOT_RUN。

新入口 `pilot_v3.execute` 复用 v2 的基础合同校验，再检查审核矩形走廊内的有限横切和来源必填 schema；收据绑定 admission-v3。通过后复用原 v1 固定数值计数器。旧收据不能直接沿用。不应再把 v1/v2 入口当当前准入；它们保留作历史复现实验。

完整横切：候选有限线分别与两条不可绕行边界有实际交点，参数在 [0,1]、交点严格在面内；ROI 覆盖审核走廊，a/b 入口面与区域侧参考一致。计数线可与原线不同，两个正例均通过。此条件只证明作者声明的矩形走廊抽象；未知真实空间形状、旁路和缺失边界证据拒绝，不能推定物理覆盖。

来源 source_id / revision、会话 source_id / source_revision，各自先验证必填、非空字符串 / 严格正整数，再比较一致性。重新签发模拟收据也不能接受缺失字段。模拟来源和审批不能认证真实人员或伪造来源，沿用 v2 的其它限制及 t0=0 支持范围。

```sh
python -m unittest discover -s pilots/building_boundary_v3 -p 'test_*.py' -v
python pilots/building_boundary_v3/run_revision.py
python pilots/building_boundary_v3/build_report_v3.py
python pilots/building_boundary_v3/manifest_v3.py --check
```

`baseline.py` 只读导入旧代码。`run_revision.py` 为独立自制空间参考增补通行面 / 不可绕行边界；不从事件 GT 选择线段。评分阶段才读取原手工 GT。真实最小输入和识门/连接区域/室外证据/执行器拆分继续见 v2 协议。

本次完成状态是“作者有界补修并提交新 head”；新 head 独审为 PENDING，当前不称准入完备。原 850b4f3 的 38 项独立软件测试证据与 v2 的 10 项作者浏览器证据分开记录。
