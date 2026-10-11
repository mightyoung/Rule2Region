# v3：I1-R无RGB基线与直接导出嵌套schema修订

基于84ac6127。旧protocol/中的六个文件（v2代码、测试、模板、输入导出、说明、结果）逐字节保留，与该提交blob比较相同；v3独立版本不覆盖或重解释旧结果。v3复用同16个空记录ID与来源组，不新增独立样本、图像或标签。

差异：

1. 新增独立mode `I1-R-no-RGB`。它保留I1-R的同一地图/B配置/相机配准、通用任务及中性ROI，只删除RGB字段。没有新增目标门对应、端点成员或答案。完整16条输入导出现在每条有5个mode；不是80个独立样本。
2. 全索引校验与leaf `export_inputs`共用RGB/任务/ROI嵌套schema；RGB仅允许reference、decoded_rgb_sha256，拒绝额外答案字段及允许键下的对象注入。ROI仅允许归一化数字矩形。R元数据原有白名单保留；E叶级配置也执行现有配置schema，拒绝额外字段。单个叶级导出不能取代全索引来源/分组校验，生产输入轮廓仍应由export_manifest生成。

代码最小变化见零上下文CODE_DELTA_FROM_V2.diff。DEV_TEMPLATES_16.json与EMPTY_PROFILE_EXPORTS_16.json均是v3新产物；旧v2文件不变。未定归属unknown，图像证据unreviewed，5层决策null，split=null，训练/效果表为空；S/B业务定义待确认。

RED/GREEN记录：初始41项测试在v2拷贝上失败（7个失败条目含子测试、2错误），复现缺少R基线与嵌套RGB/ROI透传。另加E配置叶级负测后42项中2个子测试失败。修复后42项全部通过，0失败/跳过。原v2的34过冻结结果不变。测试使用空模板和玩具软件上下文，未读取/渲染图像、推理或训练，不是视觉结果。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s research/synthetic-counterfactual-method-20261011/protocol-v3 -v
PYTHONDONTWRITEBYTECODE=1 python3 research/synthetic-counterfactual-method-20261011/protocol-v3/protocol.py --validate research/synthetic-counterfactual-method-20261011/protocol-v3/DEV_TEMPLATES_16.json
```

同上下文测试用非空玩具地图/B/配准引用验证逐字段相同、只去RGB且不修改原对象；嵌套答案负测直接调用leaf，不靠全索引提前拒绝。I1-E及其no-RGB基线仍是配置执行XOR，不算视觉发现。

这些是结构校验，不认证引用真实可用、地图没有嵌入目标帧/答案，或来源声明正确。后续真正使用R基线还需审阅实际上下文内容、运行时来源和可达性；当前资源全部未填，不声称完成真实无RGB实验。Library401继续阻断实际图审阅；未重传、未扩量、未训练。
