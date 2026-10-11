# 四帧准备索引到 f23 学习合同的只读适配

基线固定 `f23d587de0e8d82b4a823173359d3f34b0990d77`。本目录是独立 companion，不修改 `learning/`、schema、标签枚举或原16图研究协议。只映射已有私人四图预标条目和执行元数据准入检查；不读像素、不生成特征、不导入torch、不训练、不下载、不传媒体。

## 字段映射

| 私人准备字段 | 现有 boundary-v1 对应 | 保留边界 |
|---|---|---|
| sample_id | id | 原始dataset/sequence/camera/frame身份不变 |
| source_file SHA、views | input.views SHA和私有asset_ref | SHA不是访问权限或GT；公开投影不提供asset_ref |
| 来源URL/许可/日期 | source | 正式permission_status仅显式记录，不从描述字符串猜verified |
| dataset/sequence/camera | 旁路source_component、camera | f23没有camera字段；不向其input塞新字段 |
| door_and_passage_candidates | input.door_candidates normalized xyxy | 模型门检测未发生；门/通道AI种类在provenance旁路保存 |
| regions/region_pairs/view_pairs | 原同名input引用 | 未知背侧仅候选描述，不造真实拓扑；四帧view_pairs为空 |
| building/layout/door/variant/plan、split | identity、split | 缺失保留null；模板暂不符合正式manifest，不造占位ID |
| AI标签、风险、版本 | 已知AI revision时用现有ai_preannotation | AI revision未知时正式字段为null，草稿标签与风险留provenance；不借用person模型revision |
| review/truth | 原同名字段 | pending与truth=null保持，不从AI/default-adopted推人工确认 |

`adapt_preparation()` 输出的是索引，含 `boundary_v1_template` 和独立provenance。它不是新的训练协议，也不是有效正式manifest；索引本身从不宣称supervision_ready。现有 `learning.contract.validate_manifest()` 仍是完整正式合同校验器。

`admit_supervision()` 先检查人工审核、已声明分组、原输入指纹/版本/source、已声明split和来源序列不可跨split，再调用原f23校验器检查证据、权限、标签等。只有该显式检查通过才返回标签索引，没有数据加载、训练或写入。索引、审核者字符串及源哈希仍是声明/绑定，不能自动认证人审真实发生或物理几何正确。

## 实际四图结果

`REAL_FOUR_SAMPLE_INDEX.json` 为去敏投影：ChokePoint P1E_S1/C1，帧0、142、262、312，4个不同原文件SHA，同1个来源序列/相机组件，split全部null。0/1/2/1是已有person模型保留框数，不是人数GT；IDEA-Research/grounding-dino-tiny revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`，person.，box/text0.25、NMS IoU0.7。

人员模型来源revision已知，门/通道目检AI revision未知，分别记录。两类候选region关系AI均unknown，AI证据充分性暂为insufficient，全部human review pending、truth不可用；内外/建筑归属unknown。不能把这些AI关系监督为not_connected，也不能把AI insufficient当已审核标签。全部4图实际调用准入函数均被拒绝；正式监督样本数0。

源素材许可沿用已保留的NICTA ChokePoint非商业研究记录与署名；源页为 https://arma.sourceforge.net/chokepoint/ ，出版者记录 DOI 10.5281/zenodo.815657。本公开索引不发布图像、door坐标、person框/分数、原始张量或私人路径，不扩大媒体发布权限。已有原图/模型输出字节完整性另在私包核验通过；原包未修改。

同序列守卫比仅凭不同媒体SHA/可重命名building-ID分组更严格：即使各帧被填上不同分组名字，同一dataset+sequence仍不能跨split。它不能发现错误来源声明或全部近重复来源，不能代替来源审计；四张开发暴露帧不构成独立train/test。

## 软件验证

先写20个行为测试并观察20个预期失败（适配行为不存在），再实现。追加split绑定测试先观察1个预期失败，再修正。最终23项adapter测试全部通过。正例是明确自制的虚构审核元数据，仅检验合同和标签编号，不是任何真实人审。经审核的unknown按原枚举保留为第三类索引2；known connected按原枚举为1。待审、高风险/普通默认采用、无证据、失配版本、占位分组、输入修改、身份/分割修改以及同源跨split被拒绝。

本次使用不含torch的标准库解释器执行原learning全套：16 pass / 0 fail / 4 dynamic skip；这是刻意避免任何软件optimizer步骤的本轮范围，不代表此前torch环境消失。f23此前torch2.8.0+cpu的20/20动态核验已单独记录于研究交接分支提交 `096da2ee381683d51475b43c9b6799cf2068addf`，本轮不重复、不混计。原23文件learning manifest仍通过、SHA不变。

```sh
python3 -m unittest discover -s learning_adapter/tests -v
python3 -m unittest discover -s learning/tests -v
python3 -m learning.manifest
```

调用示例（已授权私人准备文件，不含下载入口）：

```python
import json
from learning_adapter.admission import adapt_preparation, public_index, admit_supervision
prepared = json.load(open("PRIVATE_PREPARATION.json"))
index = adapt_preparation(prepared)
metadata = public_index(index)
# 真实四图尚缺审核/分组，不调用训练。若未来已实际审核，先核验正式候选：
# admitted = admit_supervision(FORMAL_REVIEWED_MANIFEST, index)
```

本轮有软件/索引适配结果，仍没有真实训练或模型效果结果；素材交付认证阻断未解决，未重试Library或尝试替代上传渠道。

## I1独审修复：子集调用前核验完整索引

初版提交 `a8eeb1d8363e6ce37ebda7eb06c220d0724d1e1a` 只检查manifest选中的行。同index中同源帧声明train/test，分别单样本调用曾各自返回标签，合并调用才拒绝。此行为已用两个单样本调用RED复现；同源未分配邻帧被遗漏的第三项RED也复现。初版23/23结果和原回执保留，不能据此宣称覆盖了此漏洞。

现在先遍历完整index的所有声明，检查每个dataset+sequence组件的已分配split一致性；任何已声明跨split冲突在子集准入前拒绝。选中组件若有任何null未分配成员，也拒绝该组件的监督子集。未分配的其他组件不进入标签输出，不妨碍一个独立、完整一致的已分配组件。已有参数/返回值、f23 schema、标签语义均不变。

新增5个回归测试，适配全套现在28 pass / 0 fail / 0 skip；原learning本轮16 pass / 4 dynamic skip，原23文件manifest仍相同。真实四图的split=null、review=pending、truth=null及公开索引未改，单图准入仍4/4拒绝；训练步骤0。详见 `I1_FIX_RECEIPT.json`。原始RED与GREEN日志仅私留，公开记录只含摘要与哈希。
