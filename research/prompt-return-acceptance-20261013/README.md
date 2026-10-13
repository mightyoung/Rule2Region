# 已返回矩阵验收与人工复核入口

2026-10-13，继续原任务，不重复请求或运行CPU/GPU矩阵。已fetch并核远端完整SHA：

| 证据 | 精确Git提交 | 验收层级 |
|---|---|---|
| CPU 9次汇总 | 2b85946e48cdc46ca780d2659ab8cffa3cb664c8 | Git实际字节与回执一致性已核 |
| GPU 9次汇总 | c09db0d147ed1d1a9426f14dfc2337ea08a8da80 | Git实际字节与回执一致性已核 |
| 19脱敏回执（9 CPU+9 GPU+1 CPU smoke） | 7151e4b321f2cda87d9c99ace729c39e16015c40 | manifest逐文件bytes/SHA、帧/模型/提示/阈值/计数/汇总对应已核 |
| 原执行器 | CPU 3502616f9eca5c99ca05871daa4479058fad6ef1；GPU 2f199f8f7f7bf5b2ed383ea2fd1596132cd1c203 | 已识别两个原执行版本；不是重新执行证明 |

[PUBLIC_AUDIT.json](PUBLIC_AUDIT.json)由离线audit_returns.py读取固定Git对象生成。3个原帧SHA、模型revision与权重SHA对应冻结manifest。CPU/GPU 9项候选数均一致；**数量一致、raw框不触图像边都不是准确率或物理底界完整性**。输入哈希相同不代表独立盲评，AI预标不等于人工GT。

## 实际已收到与未收到

已收到：两份公开汇总、19份脱敏JSON回执和脱敏manifest的实际Git字节。回执中raw_tensor_sha256只是声明的摘要，收到摘要不等于收到张量。公开回执已有候选数据由其原发布分支管理，本增量不再复制候选坐标，也不扩大原始材料公开范围。

**本执行环境尚未收到操作员原始私有CPU/GPU包**。每个完整矩阵待验69个必要文件位置：RUN_PRIVATE与MODEL_ENV_PRIVATE 2个；每trial7个×9=63（source receipt、实际stdout/stderr、raw tensor、processed input、完整probe receipt、AI渲染）；3份原图；1份原始许可。该数量是验收清单，不声称服务器缺这些文件。私有服务器资源日志、freeze/check另需回传，如不存在须如实注明。缺完整底界/门身份/闭合ROI/合法S/B/进出人工GT，这是科学证据缺口，不是可从日志恢复的文件。

本环境另有此前授权的原图/模型缓存和云端单帧预检；它们不是此次服务器原始运行包，不冒充已独立验收服务器原图/张量/日志。Mac落盘仍需实测确认。

## 先做不依赖私有包的离线审计

一次联网fetch已完成。复制到离线机器前保留这三个Git对象，避免只复制当前分支遗漏结果对象：

```bash
# 仅未取得对象时由有权限的用户fetch；审计脚本本身无网络。
git fetch origin refs/heads/task/cpu-prompt-results-20261012 refs/heads/task/gpu-prompt-results-20261012 refs/heads/task/prompt-receipts-sanitized-20261012
python3 research/prompt-return-acceptance-20261013/audit_returns.py \
  --repo . --out /tmp/rule2region-public-return-audit.json
python3 -m unittest discover -s research/prompt-return-acceptance-20261013 -p test_acceptance.py -v
```

测试使用已有公共字节与合成篡改；不是模型效果、GT或新矩阵实验。原代码、原协议、已回传分支及main不覆写。

## 私有原始包离线验收（不重跑模型）

原包经已授权私有附件/原私有站点交付后，用户在私有目录核SHA并审查归档成员，按既有RUNBOOK解包。不要盲extract不可信归档；本工具只读已解包目录，不自动下载/解包、执行日志或模型forward。位置可不同，用参数指定即可。

```bash
# 使用原CPU固定环境：包含torch/safetensors，校验张量无需GPU。
PY=/absolute/private/venv/bin/python
REPO="$PWD"
PRIVATE_RETURN=/absolute/private/returned-package
"$PY" research/prompt-return-acceptance-20261013/check_private_return.py \
  --repo "$REPO" --run "$PRIVATE_RETURN/cpu-matrix" --run-name cpu-matrix \
  --source-data "$PRIVATE_RETURN/data" \
  --license "$PRIVATE_RETURN/data/SOURCE_LICENSE_ORIGINAL.html" \
  --out "$PRIVATE_RETURN/cpu-acceptance-v1"
# GPU包在同CPU环境检验raw CPU tensors，不请求GPU：
"$PY" research/prompt-return-acceptance-20261013/check_private_return.py \
  --repo "$REPO" --run "$PRIVATE_RETURN/gpu-matrix" --run-name gpu-matrix \
  --source-data "$PRIVATE_RETURN/data" \
  --license "$PRIVATE_RETURN/data/SOURCE_LICENSE_ORIGINAL.html" \
  --out "$PRIVATE_RETURN/gpu-acceptance-v1"
```

可加`--model /absolute/private/pinned-model`核9模型文件，但模型缓存不是要求回传的实验产物；没有它时明确“权重字节未验”，不把回执中的权重摘要当独立核验。输出必须在实际Git checkout外、使用新目录。

退出0：原图/许可/receipt/raw SHA及原执行器tensor/query/产物hash/矩阵检查通过，并在临时副本重生成汇总与已返回Git汇总逐项一致。原私有包不修改。需临时再预留一个矩阵副本的空间；临时副本只用于校验，结束清理。退出2：缺件，仍生成逐项缺失报告与待人审模板。哈希/范围/参数不一致抛错，保存PRIVATE_FAILURE，不改已有证据或补写GT。缺私有包时工具不导入torch、不执行张量校验，模板仍能依据已核公开回执生成。

输出PRIVATE_INTEGRITY_REPORT、HUMAN_REVIEW_TEMPLATE_PRIVATE及失败明细**均只私有保留**。原始argv/路径、图像、张量、资源日志、许可或人审坐标不入公开Git。验收仅建立文件完整性与内部对应，不保证诚实执行或准确ROI。若原包从未生成原图/资源日志/许可拷贝，请回传能合法取得的源副本并标“后补源材料”，不能补造原运行记录。

## 下一项人工复核

工具为单矩阵13个AI候选生成私有模板（不是13扇物理门）。推荐CPU作为首次人审主视图，GPU同场同提示作数值对应附件，不做第二份独立GT。步骤详见[HUMAN_REVIEW.md](HUMAN_REVIEW.md)。C1已有左右派生图贴合确认、顶部/底部证据图threshold_not_visible，保留其限定作用域；不自动扩展到C2/C3/别的提示候选，更不再问同图不可见门槛位置。

先验收原图/渲染后才请用户在原授权私有页面复核。模板未完成前review_status=unreviewed、坐标null、全闭合确认false，S/B/事件/轨迹/方向unknown。方向变化只是证据，模糊目标才调用复合研判，unknown不等于安全。无新增模型调用/训练/付费或用户服务器连接。
