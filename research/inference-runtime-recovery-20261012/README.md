# 原模型推理环境恢复与新前向（2026-10-12）

本次不止重放历史张量：按既有环境记录恢复依赖后，在真实原图上实际完成两次新的模型前向。只用已授权缓存GroundingDINO tiny同一revision；不下载新模型、购买/配置密钥、训练或改原素材/标签/协议。

## 依赖恢复已完成

在可保留的工作区创建隔离venv，不再依赖丢失的临时目录。Python3.12.14，CPU torch2.8.0+cpu、transformers4.57.1、safetensors0.6.2、tokenizers0.22.2、huggingface-hub0.35.3、numpy2.5.4、Pillow12.3.0与历史版本一致。恢复最小推理依赖，不宣称重建全部历史研究环境。

26个wheel共241278725字节，venv约1.1GiB；资源足够。CPU torch来自官方download.pytorch.org CPU索引，其余来自官方PyPI。原13项附加包按旧hash验证；全26项保存版本/文件SHA并用--no-index、--require-hashes离线安装。首次附加包命令早于pip初始化完成、首次离线安装缺Python>=3.12要求的setuptools，两份失败日志均保留；补入历史setuptools78.1.0后pip check通过。没有自动改为未锁定最新版。

[完整锁](REQUIREMENTS_RESTORED.lock)、[CPU torch锁](CPU_TORCH.lock)、[非torch依赖锁](PYPI_DEPENDENCIES.lock)与[wheel清单](WHEEL_MANIFEST.json)支持本次Linux x86_64/Python3.12重建；不是Mac或其他平台的通用wheel锁。wheel/venv本体不上传Git。

## 真实新推理证据

- 源帧142，固定person.、box/text阈值0.25、原图800×600、PIL慢processor、CPU float32、4线程/1 interop线程、seed0、无预热。新前向7.0666秒，1个person候选。新原始logits与框逐元素相等、非有限掩码相等，token ID与处理形状相同；分数、后处理坐标差均0。见[新前向回执](NEW_FORWARD_RECEIPT.json)。这是单次运行时间，不是吞吐基准或精度。
- 源帧0无遮挡，预先固定door. doorway.，相同模型与0.25阈值，未看结果调prompt/阈值。新前向7.2765秒，2个候选保留：q000分数0.4736，定位后侧局部门位；q001分数0.4204，覆盖大部分近侧开口。文字类分别door doorway、doorway，按原文本类执行IoU>0.7 NMS、没有合并同义词。候选数不等于物理门数。见[门口小试回执](DOOR_PROBE_RECEIPT.json)。不同prompt，不与旧person输出做等价复现声明。

模型snapshot9文件逐项大小/SHA验证，包括原权重689359096字节。加载使用local_files_only=True、token=False、trust_remote_code=False，并设置HF/Transformers离线。新的原始张量、处理输入和门框诊断图私下另存；Git只含代码、锁和汇总，不含图像、ZIP、模型或张量。

S的可见局部门口候选有所增加，但完整围护体积/物理门身份仍未人工确认，S图像关系unknown。B的第一版业务范围已由用户确认，见[政策增量](../first-counting-scope-decision-20261012/README.md)；实际区域/门归属仍unknown。模型输出继续AI预标待人工复核，不能把大开口候选直接当计数门，不能用模型person检测代替进出/全楼存量标签。事件、身份、轨迹与方向不新增已知结论，训练和GT效果仍为0/null。

## 可复现范围

[真实前向runner](one_real_forward.py)逐项核缓存snapshot及原图hash再调用模型，与历史张量比较；[局部门口runner](one_door_probe.py)固定第二种prompt、保存新候选。先在隔离venv中从官方CPU索引下载torch==2.8.0+cpu，从PyPI按PYPI_DEPENDENCIES.lock下载其余wheel，再按REQUIREMENTS_RESTORED.lock离线hash安装并pip check。runner参数都为调用者已获授权的实际文件路径，运行--help可见；不自动获得私有原图/缓存，不缺料时补造，不自动下载模型。公开代码本身不等于完整像素材料已交付。

历史四帧张量重放仍是旧缓存数值验证，与本次2次新前向分开。原v1图像、原四组unknown、v2/v3冻结模板及独验结果不改；新模型证据不自动并入旧标注。Library历史401保留，未重传或换路，未宣称新诊断图或原素材已保存Mac。

依赖重建命令（在本目录；只下载包，不下载模型）：

```sh
python3 -m venv recovered-venv
recovered-venv/bin/python -m pip --isolated download --index-url https://download.pytorch.org/whl/cpu --only-binary=:all: --no-deps --require-hashes --dest wheelhouse -r CPU_TORCH.lock
recovered-venv/bin/python -m pip --isolated download --index-url https://pypi.org/simple --only-binary=:all: --no-deps --require-hashes --dest wheelhouse -r PYPI_DEPENDENCIES.lock
recovered-venv/bin/python -m pip --isolated install --no-index --find-links wheelhouse --require-hashes -r REQUIREMENTS_RESTORED.lock
recovered-venv/bin/python -m pip check
```
