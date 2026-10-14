# 可照做执行入口（Linux x86_64）

CPU服务器正式批量任务优先使用 [CPU_SERVER.md](CPU_SERVER.md) 与 run_cpu_server.sh：4核、单任务串行九次、GPU可选，资源/时长的实测与估计分别列明。

本包是**三机位×固定提示词候选敏感性对照**，不是完整ROI效果实验。9次前向，无训练、付费服务或新标注。GPU可选；现有CPU已能执行。请只在你获授权、非商业使用的算力环境运行。

## 1. 检出任务分支，固定实际提交

```bash
git clone --branch task/counterfactual-method-v1-20261011 https://github.com/mightyoung/Rule2Region.git
cd Rule2Region
git rev-parse HEAD
TASK=research/gpu-prompt-boundary-handoff-20261012
PRIVATE_RUN="$HOME/rule2region-private-prompt-run"
mkdir -p "$PRIVATE_RUN"
```

PRIVATE_RUN 必须在所有实际git checkout外，新目录不得覆写。不要用 `$HOME` 本身作为运行目录；所有数据/模型/输出只进PRIVATE_RUN。记录当前完整SHA，回传中 git_commit 对应实际运行版本。首次 clone 后如需复現指定版本，使用交接最终报告的完整SHA checkout（不改main）。

## 2. 环境固定与资源

Python **3.12.14**、torch **2.8.0+cpu** 或 **2.8.0+cu128**，transformers4.57.1、tokenizers0.22.2、safetensors0.6.2、huggingface-hub0.35.3、NumPy2.5.4、Pillow12.3.0。先用本机/管理员已有Python3.12.14建立venv；本包不自动安装系统Python、驱动或启动GPU。

```bash
python3.12 -m venv "$PRIVATE_RUN/venv"
PY="$PRIVATE_RUN/venv/bin/python"
"$PY" -c 'import platform; assert platform.python_version()=="3.12.14"'
# CPU预检或CPU全矩阵：
"$PY" -m pip install --no-deps --require-hashes --index-url https://download.pytorch.org/whl/cpu -r research/inference-runtime-recovery-20261012/CPU_TORCH.lock
"$PY" -m pip install --no-deps --require-hashes -r research/inference-runtime-recovery-20261012/PYPI_DEPENDENCIES.lock
# 若运行CUDA，替代上面两条CPU安装命令：官方CUDA 12.8构建，CUDA依赖由此精确torch版本声明固定。
# "$PY" -m pip install 'torch==2.8.0+cu128' --index-url https://download.pytorch.org/whl/cu128
# "$PY" -m pip install --no-deps --require-hashes -r research/inference-runtime-recovery-20261012/PYPI_DEPENDENCIES.lock
"$PY" -m pip check
"$PY" -m pip freeze > "$PRIVATE_RUN/environment.freeze.txt"
```

原CPU轮子哈希锁针对Linux x86_64/CPython3.12，其他平台哈希不匹配即停，不删除hash绕过。本包GPU安装固定torch构建及顶层依赖；**CUDA轮子与全部CUDA传递依赖未在本环境下载/测量，不声称GPU完整wheel哈希锁已验证**。保存实际freeze、下载轮子SHA与驱动日志私有回传；仅环境满足固定版本且CUDA可用才前向。官方构建参考[PyTorch历史版本](https://pytorch.org/get-started/previous-versions/)。驱动需支持CUDA12.8，管理员按官方兼容性核验；不自动升级。

建议预留 **8 GiB GPU显存、16 GiB RAM、15 GiB磁盘**；这是保守准备预算而非实测最低值。模型权重约689 MB，源归档385 MB，三个JPEG约159 KB，9次输出（处理输入张量约17 MB/次）预留1 GiB。CUDA安装包/缓存占主要磁盘；不同硬件未实测。记录 `nvidia-smi > "$PRIVATE_RUN/nvidia-smi.txt"`。batch1、float32、TF32关闭、无warmup，实际peak allocated显存写receipt；OOM立即停，保留日志，不调分辨率/精度后冒充同实验。

## 3. 合法原图获取（下载或导入已有完整归档）

唯一公开数据源为[作者ChokePoint发布](https://zenodo.org/records/815657)，使用P1E_S1.tar.xz，**不下载face crops/GT/其他视频**。CC BY NC 4.0，保留[官方许可条件](https://arma.sourceforge.net/chokepoint/)及NICTA/Wong等2011 CVPRW署名，非商业，标注衍生材料。本实验并非人脸识别或身份验证。

```bash
curl --fail --location --proto '=https' --proto-redir '=https' \
  https://zenodo.org/api/records/815657/files/P1E_S1.tar.xz/content \
  --output "$PRIVATE_RUN/P1E_S1.tar.xz"
curl --fail --location --proto '=https' --proto-redir '=https' \
  https://arma.sourceforge.net/chokepoint/ \
  --output "$PRIVATE_RUN/SOURCE_LICENSE_ORIGINAL.html"
"$PY" "$TASK/handoff.py" prepare \
  --archive "$PRIVATE_RUN/P1E_S1.tar.xz" \
  --license "$PRIVATE_RUN/SOURCE_LICENSE_ORIGINAL.html" \
  --data "$PRIVATE_RUN/data"
```

也可把已合法保留的同SHA完整原归档和原许可HTML复制到这两个路径，不需要再次下载。prepare读取精确nested/member名称、不用extractall、不执行归档内容，校验原归档及每JPEG SHA，保留许可副本。若官方页面更新导致许可HTML哈希不同、403或网络失败，**停止**：改用已授权私有保存的原通知副本；不能换镜像、绕过许可或把私有站点链接加入公开manifest。原许可固定哈希见prepare；本环境实际导入该副本成功。没有该合法副本时只阻断prepare与实测，合成测试仍可跑。公开输入manifest包含源帧哈希及有限复核状态，没有图片或私有坐标。

## 4. 模型获取（已有缓存可直接复制）

只允许`IDEA-Research/grounding-dino-tiny` revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`。9文件名/字节/hash在model_manifest.json；权重SHA `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`。作者模型许可Apache2.0，保留模型卡和原归档中license信息，禁止替换latest。下列命令是**由用户在自己授权资源上显式执行的下载**，本次助手没有执行或下载权重。

```bash
"$PY" "$TASK/download_model.py" --model "$PRIVATE_RUN/model"
```

也可复制既有完整9文件缓存到model；下载命令不是run的隐式依赖。run强制离线、local_files_only、token=False、trust_remote_code=False，无密钥需求。

## 5. CPU预检与正式实验

```bash
"$PY" -m unittest discover -s "$TASK" -p test_handoff.py -v
"$PY" "$TASK/handoff.py" run --data "$PRIVATE_RUN/data" --model "$PRIVATE_RUN/model" \
  --out "$PRIVATE_RUN/cpu-smoke" --device cpu --smoke
"$PY" "$TASK/handoff.py" validate --out "$PRIVATE_RUN/cpu-smoke"
# GPU正式9次前向（使用CUDA venv）；CPU全矩阵将cuda改为cpu即可。
"$PY" "$TASK/handoff.py" run --data "$PRIVATE_RUN/data" --model "$PRIVATE_RUN/model" \
  --out "$PRIVATE_RUN/gpu-matrix" --device cuda
"$PY" "$TASK/handoff.py" validate --out "$PRIVATE_RUN/gpu-matrix"
```

CUDA环境不能运行固定cpu-build smoke检查；CPU smoke使用单独CPU venv，上述官方CPU锁创建另一个venv即可。或先在本机CPU执行预检，再在CUDA机器执行9矩阵。不要覆盖前次输出；出现失败用私有失败记录返回，不自动重试。

基线combined=`door. doorway.`；对照door_only=`door.`与doorway_only=`doorway.`。唯一主要变量是提示词，机位分层比较，不把跨机位帧注册或认成同一物理门。固定.25/.25、same-exact-label NMS IoU>.7、seed0、full RGB、无掩膜、原尺寸800×600。可看候选数、raw框截边率及同机位提示变化；不能从候选更多或更稳定推导准确率。CPU/GPU只可比较运行/数值差异，不要求bytewise一致。所有AI候选待逐项人审，轨迹、方向、S/B及事件始终unknown。方向变化只是证据，模糊目标才使用复合研判。

停止条件：源/模型/环境不匹配、GPU不可用、OOM、子进程非零、张量非有限、结果验证失败、许可拒绝；保留实际日志和unknown。缺完整底界/人工GT时停止完整ROI解释，当前批即属于这一范围。

## 6. 输出验证与回传

详见RETURN_SCHEMA.md及public_result.schema.json。验证器检查9试验矩阵、原始tensor/query对应、产物哈希、scope及参数漂移。输出目录实际argv、PID、UTC/monotonic、stdout/stderr、原始/输入张量、PNG都私有。合成测试只证明软件契约。

```bash
# 上一步validate成功才会产生PUBLIC_AGGREGATE.json。
# private目录包含许可与所有输出；gzip包仅通过已批准私有站点或私有附件回传：
"$PY" research/prompt-return-acceptance-20261013/pack_private_return.py \
  --root "$PRIVATE_RUN" --run-name gpu-matrix \
  --out "$PRIVATE_RUN/gpu-private-results-v2.tgz"
sha256sum "$PRIVATE_RUN/gpu-private-results-v2.tgz" > "$PRIVATE_RUN/gpu-private-results-v2.tgz.sha256"
# 公开回传分支，只copy允许的汇总，不git add整个private目录：
git switch -c task/gpu-prompt-results-20261012
mkdir -p research/gpu-prompt-boundary-handoff-20261012/returns/operator-run-1
cp "$PRIVATE_RUN/gpu-matrix/PUBLIC_AGGREGATE.json" \
  research/gpu-prompt-boundary-handoff-20261012/returns/operator-run-1/aggregate.json
git add research/gpu-prompt-boundary-handoff-20261012/returns/operator-run-1/aggregate.json
git diff --cached --check
git diff --cached
git commit -m 'Return validated fixed-prompt candidate sensitivity aggregate'
git push -u origin task/gpu-prompt-results-20261012
git rev-parse HEAD
git ls-remote origin refs/heads/task/gpu-prompt-results-20261012
```

运行ID/分支已存在时选新名字，不强推。检查暂存文件只有汇总JSON且无本机路径、秘钥、私有URL、坐标/图像。失败时不运行上述成功汇总命令；私有回传失败RUN/logs，公开只手写审阅后的错误类别、unknown及未完成次数。Mac落盘仍须用户实测确认。

2026-10-14：新打包入口还强制加入matrix之外的三张data/C{1,2,3}-f00000000.jpg原图（逐SHA核验），避免旧tar命令漏列。只新建私有归档并记录成员SHA，不覆写旧包/原日志/张量；清单齐全不等于包内容已验收。缺图会拒绝，显式--allow-partial仅允许保存已有失败矩阵产物和缺件记录，原图/许可仍必需。原图只私有回传，绝不提交公开仓库。不据此推断旧服务器包实际缺图。
