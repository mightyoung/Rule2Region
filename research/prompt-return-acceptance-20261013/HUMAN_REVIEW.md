# 私有人工复核清单：不补造看不见的底界

先确认 PRIVATE_INTEGRITY_REPORT 的原图SHA和具体source_id，打开私有原图与对应AI_PENDING_HUMAN图。保持原图原尺寸，不使用增强/裁剪替代原证据，不把图片下边或候选框下边当作物理门槛。记录看过AI=true；这种复核不是盲GT。

1. **视野是否充分**：分开记录top/left/right/bottom可见性（visible、not_visible、occluded、unknown）。需要时保持unknown。不可见阈值只标threshold_not_visible；不能解释为物理不存在、负类或安全。完整开口所需边界缺任何一项，full_opening_geometry_confirmed保持false。
2. **候选指向什么**：按source_id、arm、prediction_id逐项复核，区分关门板、近侧被截开口、室内不同深度开口、其他/无法判断。AI框不是开口多边形，候选数不是门数；不同机位没有注册或已确认物理门身份。
3. **局部确认可独立保存**：可确认可见左/右沿贴合，同时顶部/底界unknown。C1此前左右派生图确认与threshold_not_visible保留原范围，不覆写成当前13候选全部已确认，不迁移给C2/C3或新提示词。不要反复要求用户标同图看不见的位置。
4. **门身份与政策分开**：单栋封闭室内的B政策已确认，但当前帧的建筑/房间/合法地图连接仍unknown。关门板或房间入口不能自动成为该栋计数边界。无图像对应/map时保持S/B unknown。
5. **计数与轨迹**：本次只有三个背景单帧，不支持从单帧推导轨迹/方向/进出。全部保持unknown；后续合法连续证据另行匹配，方向变化只是证据，模糊目标才复合研判。
6. **保存真实审阅过程**：填reviewer、真实UTC、viewed_original_sha256、seen_AI_candidates、具体note；只有实际复核后才改partial_review/needs_more_evidence等状态。不默认接受AI，不“AI预标转人工GT”。需要画可见局部时坐标单独私有保存，未见点保持null。

第一轮优先检查C2的combined小框是否关门板、近侧大框是否仍下界截断；C3 door_only增加的第二候选指向何处。它们是已有提示敏感性结果的区分性人工检查，不新跑矩阵，不预设哪一候选正确。所有结论必须由对应原图/用户审阅产生，当前只是待复核清单。

资料仅进入原授权私有站点或私有附件；不能把人审内容/原图发布GitHub。若尚缺私有包，停在待复核入口，明确缺件，不向用户展示由本工具伪造的“已审图”。

## 填表后离线合同检查（2026-10-13增量）

保留原HUMAN_REVIEW_TEMPLATE_PRIVATE.json，复制为新的私有review文件后填写，禁止覆写原模板或原站点记录。原图实际尚未收到时不填任何人工确认。校验器允许局部可见边沿复核；完整ROI、物理门身份、门槛定位和事件尚未获本批证据准入，均保持unknown/false。原C1派生图既有确认不由此工具更改，也不批量迁移为13条新确认。

```bash
python3 research/prompt-return-acceptance-20261013/validate_human_review.py \
  --repo "$PWD" --run-name cpu-matrix \
  --template /absolute/private/HUMAN_REVIEW_TEMPLATE_PRIVATE.json \
  --review /absolute/private/HUMAN_REVIEW_FILLED_PRIVATE.json \
  --integrity-report /absolute/private/cpu-acceptance-v1/PRIVATE_INTEGRITY_REPORT.json \
  --out /absolute/private/NEW_REVIEW_CONTRACT_REPORT.json
```

工具核固定Git回执生成的原始模板、不可变AI/原帧身份、候选集合、实际UTC/审阅者/原图SHA、AI辅助暴露状态；不可见或未知边界禁止坐标，图片底边不能代替门槛。所有字段未改的模板可通过，报告明确列未审数，**通过合同校验不等于已人审或审阅内容为真**。原图/底界/GT及原始包缺口继续保留。输出只新建、必须在真实git checkout外；不写私有站点、不发布、不执行模型。GPU模板用--run-name gpu-matrix，不能拿CPU模板冒充GPU模板。

新增人工填写的接收门槛：任何partial_review/needs_more_evidence记录必须提供同一矩阵的PRIVATE_INTEGRITY_REPORT.json，状态为private_bytes_and_original_validator_checks_passed、missing为空且raw/processed tensors已验。缺原图/许可/原始张量、运行失败或错矩阵时停止接收已审记录；不能手改报告伪装通过。全未审的原始模板仍可不提供报告作合同检查，但报告明确为未审、没有人工确认。报告门槛核软件流程声明，不是签名证明或审阅内容真实性证明。原13条AI字段与图像SHA仍不可变，完整ROI/GT/事件/门槛位置仍禁止升级。人工审阅完成后的私有回传应同时附原模板、新填表文件、对应完整性报告与新合同报告；不上传GitHub，不改写既有站点确认。
