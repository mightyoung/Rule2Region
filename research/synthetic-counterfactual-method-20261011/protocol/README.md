# 16条空开发模板及软件校验

标准库Python实现，无媒体、模型、联网或安装。详见上层PROTOCOL_REVISION_V2.md；不修改f23接口。

- DEV_TEMPLATES_16.json：完整16记录索引。源几何/政策归属unknown、image_evidence=unreviewed、decision全null，split=null。
- EMPTY_PROFILE_EXPORTS_16.json：同一16条记录的四种输入轮廓（不是64个独立样本）。sample_id仅是连接键；算法只读所选profiles子对象，不把ID、family、source/group信息当特征。I1-E与no-RGB基线引用同一配置语义；基线无RGB字段。
- protocol.py：模板构造、全索引校验、来源连通分组、输入白名单导出、无RGB依赖的配置XOR。None/unknown端点保持unknown，不能变成False。
- test_protocol.py、VALIDATION_RECEIPT.json：软件负例与回执，不是视觉评价。

从仓库根执行：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s research/synthetic-counterfactual-method-20261011/protocol -v
PYTHONDONTWRITEBYTECODE=1 python3 research/synthetic-counterfactual-method-20261011/protocol/protocol.py --validate research/synthetic-counterfactual-method-20261011/protocol/DEV_TEMPLATES_16.json
```

`--write-template DESTINATION.json`只创建开发空模板；会覆盖所指定输出，不应指向已审阅文件。export_manifest必须先验证完整16记录索引，再导出轮廓，不能用选中子集掩盖来源连接。全部布局变体/双侧在来源族中；不同族如果共享祖先、解码RGB或v1原组就合并。单一v1祖先组的身份原样保留；多个v1组因复用连通则合并，祖先注册表保留原始组关联。未声明的借用或伪造来源本身无法凭本协议发现，需要独立来源审计。

输入可用性在本版本是对声明引用/版本/审阅状态的结构核对，不是文件存在、真实性或运行时可达性证明。I1-R资源要求runtime_available、confirmed和非答案标记；严格拒绝显式目标端点答案字段，但不会打开资源内容，不能证明外部地图没有答案泄漏或模拟器真值。真正使用前仍须审阅实际资源来源、内容及可达性。RGB hash同样是声明，本版不读取图像核验像素。当前所有引用为空，0已审图、0决策、0训练/效果行。

源图真值不能流入input_available或图像证据。中性ROI只允许归一化矩形，不带S/B标签；地图/B配置应对应现实可得输入，不能隐藏导入模拟器字段。显式端点成员只能走I1-E配置执行，包含unknown时不算XOR已知结果。当前软件测试里的玩具配置/哈希不表示真实场景、文件或人工标注。
