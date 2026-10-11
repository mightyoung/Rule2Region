# Boundary learning v1

[计划 v1](plans/2026-10-11-boundary-learning-v1.md) 与 [站点标注交接](annotation-guide.md) 是本轮入口。代码是最小软件原型：给定外部候选/冻结特征，学习无方向门—区域对、视图门对和区域对证据充分性；不实现门检测、真实视觉编码器适配、完整图 cut、几何准入、计数器或站点。

从仓库根目录运行，不需修改原 `pyproject.toml`。正式数据为空，既无训练结果也无已完成 AI 真实预标注。

```bash
python -m learning.validate learning/data/real_manifest.json
python -m learning.validate learning/data/software_manifest.json
python -m unittest discover -s learning/tests -v
python -m compileall -q learning
```

没有 PyTorch 的环境会明确跳过 shape/梯度/冻结与 resume 测试。只在已有本地 PyTorch 的环境可运行下列 **软件测试**，不下载权重：

```bash
python -m learning.train learning/data/software_manifest.json learning/data/software_features.json --software-test --steps 2 --checkpoint /tmp/boundary-software.pt
python -m learning.train learning/data/software_manifest.json learning/data/software_features.json --software-test --steps 4 --checkpoint /tmp/boundary-software.pt --resume
```

合成测试的 loss 只检验梯度路径，不是模型效果。真实运行必须换用经过验证的 `real_research` manifest 和冻结特征，不传 `--software-test`；不能混合类型。每个 split 要有审核标签，未复核记录在正式 dataset 中失败。

## 冻结编码器与缓存

`FrozenEncoder` 接收显式提供的本地模块，只在 eval/no_grad 下调用；模型特定的 visual/context pooling 适配器需以后实际实现。`RelationHead(D, hidden=64)` 输入 `[关系候选数,D]` 门、区域 A/B，`[视图候选数,D]` 门 A/B；对区域/视图交换对称，输出三组 `[候选数,3]` logits。局部 ID 不进网络，unknown 是第三类。空视图候选允许，训练至少一个非空标签任务。

冻结缓存 JSON 合同的可运行例子为 `data/software_features.json`。真实缓存按相同结构：purpose=real_research；encoder 的 revision/preprocessing 精确标识，frozen/input_only=true；每样本 input_sha256 为 `contract.fingerprint(row['input'])`；doors/regions 的局部 ID 对应长度 D 的有限浮点数组。禁止 GT 特征；此自声明仍需生产代码审计。v1 使用 JSON 以便小数据审查，缓存大规模格式优化不在本轮。

## Kaggle 2×T4 后续运行条件

用户指定资源基线为两块 T4，各 16 GB，当前未访问账户、启动作业或实测。账户剩余配额/会话时限未知。官方 [T4 数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf) 标明单卡 16 GB；[PyTorch DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html) 每个进程持有模型副本，**DDP 不池化显存**，不能把两卡当作一块 32 GB。

先离线生成并核验冻结特征，再单卡 profile（真实材料/合法权重实际具备后）：

```bash
python -m learning.train LOCAL_MANIFEST.json LOCAL_FEATURES.json --device cuda:0 --steps 5 --seed 20261011 --checkpoint /kaggle/working/boundary-head.pt
```

输出实际每步 loss/step_seconds、峰值 allocated/reserved CUDA bytes，预热与后续步分开看；步耗时不包括 checkpoint 序列化。不要据头的显存推断视觉编码器或 LoRA 能装入。当前入口只支持单进程单卡，**不是 DDP trainer**；profile 后再决定实现 DDP 还是两卡独立不同种子 profile，后者各自输出独立 checkpoint。不得使用两个进程同写一个 checkpoint。500M–1B 小 VLM LoRA 是次选；3B QLoRA 仅后续实测，[PEFT 量化指南](https://huggingface.co/docs/peft/developer_guides/quantization) 是方法依据，不保证本任务可装入。

checkpoint 每步原子替换，记录权重、optimizer、step、CPU/CUDA RNG、种子、manifest/cache 指纹、结构、学习率、设备及 torch 版本。resume 要完全匹配；总 steps 表示恢复后训练至多少步，样本按 `step % len(train)` 确定性循环。测试比较中断恢复与连续执行。跨软件版本/设备复现不保证位级相同。仅加载自己可信的本地 checkpoint，`weights_only=True`；不上传 checkpoint/输出日志到公开 Git。会话结束前持久化 checkpoint 需账户实际具备合法存储，当前未配置。

## 校准与评估入口

标签索引为 relations=[not_connected,connected,unknown]，matches=[different,same,unknown]，sufficiency=[insufficient,sufficient,unknown]。unknown 与低置信拒判分别报告。下面是运行说明，当前没有真实数据/模型，未执行：

```bash
python -m learning.evaluate LOCAL_MANIFEST.json LOCAL_FEATURES.json LOCAL_CHECKPOINT.pt --split calibration --max-risk USER_FROZEN_LIMIT --policy /kaggle/working/policy-v1.json
python -m learning.evaluate LOCAL_MANIFEST.json LOCAL_FEATURES.json LOCAL_CHECKPOINT.pt --split test --policy /kaggle/working/policy-v1.json
```

阈值仅在 calibration 上拟合；test 不接受 max-risk/调阈值参数，只加载绑定 manifest/checkpoint SHA 的冻结 policy。无可行策略就失败，不能临时调 test。风险上限须用户在校准前版本化，代码没有虚构默认上限。软件模式可额外传 `--software-test`，输出始终 software_test_only；随机头不保证校准能找到策略。

评估输出为任务边际 selective risk/coverage、unknown 及描述性曲线；sufficiency 的 insufficient 是有效任务判断，但不会许可边界部署。真正 relation 接受还必须证据 sufficient、候选完整/拓扑完整、同门去重与后续准入全部通过。本轮没有 joint deployment evaluator，不将独立任务指标说成边界完整覆盖/人数漂移；这些必须按计划接入真实 GT 和固定计数器后再测。测试曲线不得用于再调策略。
