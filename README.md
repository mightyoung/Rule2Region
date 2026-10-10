# Rule2Region

业务规则 → 候选监控 ROI 与配置。研究目标是让 LLM 根据规则和固定摄像机场景提出可审查的区域建议，由人确认后交给固定检测器执行。示例场景包括消防通道杂物、室内吸烟与工作区手机使用；这些是目标场景，本仓库尚未验证其检测能力。

## 最新研究进度（2026-10-10）

[冻结16图边界支持批报告](docs/research-handoff/reports/progress-20261010-boundary16.md)：16图完成预标（14次新推理＋2张像素等同缓存复用），人工人员真值与ROI几何仍待审核，所有效果指标保持null。业务范围确认、失败/未知、证据哈希与下一步见报告。此更新位于研究交接审核分支，旧快照保留。

## 源码发布快照状态

这是从 `human-count` 提取的源码发布快照。现有工程主要是人员计数、场景配置、ROI 校验、稀疏视觉证据和失败闭合工具，保留 `codecref_lite`、`evidencegate` 等内部名称以避免破坏导入。旧的计数和配置实验不等于上述三个业务场景已经实现；当前没有可据此宣称的跨场景效果、正式科研验证或生产就绪结论。

计划中的主链路为：业务规则与场景输入 → LLM 候选 ROI/配置 → 几何与规则校验 → 人工确认 → 固定检测器执行与日志。LLM 建议不得直接变为上线配置。初期不以更换或训练检测器作为主要研究变量。

## 安装与轻量运行

需要 Python 3.10–3.13。下列安装会从包索引获取依赖；本次发布核验使用已有本地环境，没有训练或下载模型、数据。

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e . pytest
python -m codecref_lite.cli --help
python -m codecref_lite.cli select uniform --candidate-count 8 --budget 4
python -m pytest -q tests/test_selectors.py tests/test_features.py tests/test_admission.py tests/test_parsing_analysis.py tests/test_alignment.py tests/test_labels.py tests/test_roi_structure_protocol.py tests/test_evidencegate_core.py
```

这些命令验证选择器与软件合同，不会生成已完成的规则到 ROI 系统。`research_tools/qwen_business_profile_mapper.py` 是现有业务偏好映射与候选配置选择工具，依赖外部 registry/cards/catalog；它不是通用 ROI 生成接口。

## 文件范围

- `codecref_lite/`：选择器、编解码特征、数据与输入准入。
- `evidencegate/`：证据输入、资源账本、运行合同和失败闭合。
- `research_tools/`：历史开发工具源码；服务器执行子目录未发布。
- `tests/`：保留的软件测试源码；部分历史测试需要未发布的数据、研究冻结文件、远端启动脚本或证据，完整套件不是本发布包的通过承诺。
- `docs/`：发布范围、外部资产获取与本次验证记录。

服务器路径在发布副本中改为 `/srv/project`、`/srv/project-assets` 等占位路径。涉及远端执行的历史工具须自行配置合法资产与运行环境后审查使用。原始开发目录保持不变。

## 数据与研究边界

本仓库不带 `.env`、凭据、原始视频、数据集、模型权重、运行输出、私有会话或论文全文。外部资产按 [数据说明](docs/external-assets.md) 登记与获取；发布文件哈希见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。未发布的实验结果不得由源码、测试或占位配置推定。

下一步是冻结业务规则与人工确认接口，构建可追溯 ROI 候选合同，在独立数据上检验区域覆盖、配置时间、误报漏报和拒绝建议边界。

本次未新指定项目许可证；第三方依赖与数据遵循各自许可。
