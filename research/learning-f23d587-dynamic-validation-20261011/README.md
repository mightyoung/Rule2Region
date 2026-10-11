# f23d587 独立动态软件核验

被测公开源码固定为 `f23d587de0e8d82b4a823173359d3f34b0990d77`，来源 `https://github.com/mightyoung/Rule2Region.git`，原开发分支 `task/boundary-learning-plan-v1-20261011`。独立 detached checkout；本记录追加到研究交接分支，不修改被测 learning 源码或原16图协议。此处仅报告该提交，未混入 e626 的测试结果。

现存 Python 3.12.14、PyTorch 2.8.0+cpu、pytest 9.1.1；CUDA不可用。未安装依赖或下载权重。OMP/MKL一线程，HF/Transformers离线，输入仅自制8维张量和3个software fixture；真实manifest为0样本。

| 核验 | 实际结果 |
|---|---|
| learning源码manifest | 23文件通过；SHA `e22ee8ffa5c4a658a7db65c99bb73685fb6d90fb68d8ffd308e44e3cdfd78f95` |
| 全套learning unittest | 20 pass / 0 fail / 0 skip |
| 独立动态torch unittest | 4 pass / 0 fail / 0 skip；已包含在20项内 |
| 附加CLI校准正例 | 1项符合预期，exit 0；空matches阈值null、not_applicable |
| 附加CLI拒绝负例 | 4项符合预期，各exit 1；不是4个测试失败 |

动态断言覆盖shape、有限梯度、冻结encoder no_grad/eval、交换对称性、软件/真实入口隔离、连续4步与2步中断再恢复的模型参数位级相同、变更恢复seed拒绝。恢复比较没有证明完整optimizer/RNG状态等价或跨设备复现。新relation-only calibration/test CLI回归报告空matches不适用、无阈值、空描述曲线，不造接受数或风险成绩。

4个额外CLI负例：calibration无matches而test新增matches时拒绝评分；test传max-risk拒绝调阈值；把有阈值relations伪标not_applicable时拒绝；强制三类最高分并列导致无可行校准覆盖时拒绝且不写policy。known/unknown和known/known并列的保守行为亦由该提交标准库回归测试覆盖。强制logits仅为软件负例fixture，不是模型效果。

## 复现命令

只使用已有带torch的Python，不安装。以下目录名为自行选择的相对占位目录，不是原执行环境路径；不能从交接分支直接推断其已有learning源码。获取固定公开源码到独立目录，禁止切换原任务工作树：

```sh
git init fixed-source
git -C fixed-source fetch --depth=1 https://github.com/mightyoung/Rule2Region.git f23d587de0e8d82b4a823173359d3f34b0990d77
git -C fixed-source checkout --detach FETCH_HEAD
git -C fixed-source rev-parse HEAD
cd fixed-source
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m learning.manifest
python -m learning.validate learning/data/real_manifest.json
python -m learning.validate learning/data/software_manifest.json
python -m unittest learning.tests.test_torch -v
python -m unittest discover -s learning/tests -v
```

附加CLI负例的命令模板在 `CLI_REJECTIONS.json`；必须先生成与checkpoint匹配的软件fixture，而非随意修改正式数据。`reproduce_cli_rejections.py` 是实际执行探针的去路径版本，要求固定source SHA且输出目录全新、位于源码目录外。它仅执行一次软件optimizer步，然后强制假logits，检查一个校准正例和四个预期拒绝，不触发真实编码器/媒体：

```sh
python RESEARCH_CHECKOUT/research/learning-f23d587-dynamic-validation-20261011/reproduce_cli_rejections.py \
  --source FIXED_SOURCE_CHECKOUT --output-dir NEW_PRIVATE_PROBE_DIRECTORY
```

本次未因发布摘要而重复全套软件测试；上述原执行结果与回执哈希保留。探针公开版只做参数化去路径处理，语法核验通过；未再次执行该去路径版本，不把它另记成一次通过。

## 包装失败与限制

本次测试包装脚本最后的SHA断言曾因“先替换短前缀、再替换完整SHA”的顺序错误，形成错误预期字串并exit 1。此前六个子进程均exit 0，实际Git SHA独立核验正确。失败原脚本与日志私留；`WRAPPER_FAILURE_SUMMARY.json` 公开去敏原因、原脚本SHA和恢复过程。没有修改learning源码、掩盖失败或将包装失败算成源码测试失败。

`RECEIPT.json`、`CLI_REJECTIONS.json`仅提供摘要/命令/日志哈希，不发布原始日志、实际用户/主机路径、图像、person输出或checkpoint。私有四图标注包另行保留，未纳入Git。软件测试不是视觉模型效果、拓扑完整性、真实标注、占用人数或部署验证；真实训练样本仍为空。本轮没有模型权重下载、付费调用、真实模型推理或研究训练。
