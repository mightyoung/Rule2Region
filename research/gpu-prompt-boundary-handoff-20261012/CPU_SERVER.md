# CPU服务器批量执行：原定九次前向，GPU可选

原RUNBOOK已支持`--device cpu`全矩阵；此页补齐服务器资源、批量日志和CPU回传路径，不改变模型/数据/提示词/阈值/9次实验协议。云端仅开发、小预检和回传后的分析；完整9矩阵由用户在其已授权计算服务器执行。脚本不连接服务器、不申请/启动资源、不下载数据或权重。

## 资源与耗时边界

| 项目 | 配置或观测 | 证据性质 |
|---|---|---|
| CPU线程 | 4 intra-op、1 inter-op，batch1 | 现有forward.py固定值；脚本限制OMP/MKL/OpenBLAS=4 |
| 任务并行数 | **1**，9个子进程依次执行 | 执行器结构，不启多机/多任务并行；一次完整矩阵即可 |
| 服务器申请 | 4 CPU核、16 GiB RAM、5 GiB空闲磁盘；无GPU | CPU保守准备预算，不是实测最低值；源/模型重复副本及pip缓存额外占用需另算 |
| 单次模型前向 | **18.0211秒** | 当前云端CPU修正后smoke实测，C1 combined、4线程；不能代表用户服务器 |
| 单子进程完整墙钟 | **26.9377秒** | 同一次RUN_PRIVATE真实monotonic起止，含模型加载/哈希/写输出 |
| 单smoke目录 | **18,338,811 bytes** | 当前已验证smoke输出目录实际总大小；不含venv/模型/源归档 |
| 9次子进程外推 | 约242秒 | 9×26.9377简单估计，未实测完整矩阵；三提示token不同、共享机器/I/O会改变时间 |
| 调度准备时间 | 预留20分钟运行窗口 | 保守计划值，含预检/核验/打包，不含依赖/源/模型下载；不是超时重试策略 |
| 峰值RAM / 用户服务器时长 | 尚未知 | 云端未测peak RSS；服务器脚本可用GNU time采集，不编造硬件性能 |

不扩大CPU线程/实验并行数。若服务器只有少于4个可用CPU、RAM/磁盘不足，申请匹配资源或记录未运行原因，不自改实验参数。GPU路径继续保留，准备预算与未验证CUDA限制见RUNBOOK。

## 一批九次的可执行命令

先完成RUNBOOK的固定CPU venv、合法归档/许可`prepare`以及9文件模型校验。使用**CPU版torch2.8.0+cpu**，不能把CUDA venv直接拿来冒充固定CPU环境。将以下路径设为服务器上实际绝对路径，PRIVATE_RUN在所有真实git checkout外，原数据无需上传公开仓库。

```bash
# 在用户已授权服务器现有shell执行；不含ssh/sbatch或资源启动。
TASK=research/gpu-prompt-boundary-handoff-20261012
PRIVATE_RUN="$HOME/rule2region-private-prompt-run"
PY="$PRIVATE_RUN/venv/bin/python"
bash "$TASK/run_cpu_server.sh" "$PWD" "$PY" \
  "$PRIVATE_RUN/data" "$PRIVATE_RUN/model" "$PRIVATE_RUN/cpu-matrix"
```

可在既有调度系统作业中执行同一条bash命令：由用户填其队列/账户/CPU与内存字段，本包不猜调度器或提交作业。若需要与登录会话脱离，可由用户自行在已获分配的执行环境使用tmux；不要在登录节点自行启动重计算。原定9次按顺序执行，首个失败停止，原样保存unknown、错误和已生成产物，不自动重试或拆分并行。

`cpu-matrix`包含全部实验产物；独立`cpu-matrix.server-logs`保存pip freeze/check、实际批量stdout/stderr、GNU time墙钟与峰值RSS（若服务器有/usr/bin/time；无该工具则不声称测到RSS）。不覆盖既有目录。该批脚本只做9次正式前向及结果核验，没有额外warmup或重复smoke。已在云端完成的单帧预检不等于用户服务器运行成功。

## CPU结果回传

脚本返回0且validate.stdout.log包含valid=true后，按RETURN_SCHEMA回传。失败时私有回传RUN/logs，不导出成功汇总。

```bash
# 全产物私有附件/原已授权私有站点上传；绝不GitHub/LFS公开。
tar -czf "$PRIVATE_RUN/cpu-private-results.tgz" -C "$PRIVATE_RUN" \
  cpu-matrix cpu-matrix.server-logs data/SOURCE_LICENSE_ORIGINAL.html
sha256sum "$PRIVATE_RUN/cpu-private-results.tgz" > "$PRIVATE_RUN/cpu-private-results.tgz.sha256"
# 公开分支只接收严格验证器生成的无图像/坐标/私有路径汇总。
git switch -c task/cpu-prompt-results-20261012
mkdir -p research/gpu-prompt-boundary-handoff-20261012/returns/cpu-operator-run-1
cp "$PRIVATE_RUN/cpu-matrix/PUBLIC_AGGREGATE.json" \
  research/gpu-prompt-boundary-handoff-20261012/returns/cpu-operator-run-1/aggregate.json
git add research/gpu-prompt-boundary-handoff-20261012/returns/cpu-operator-run-1/aggregate.json
git diff --cached --check
git diff --cached
git commit -m 'Return validated CPU fixed-prompt candidate sensitivity aggregate'
git push -u origin task/cpu-prompt-results-20261012
git rev-parse HEAD
git ls-remote origin refs/heads/task/cpu-prompt-results-20261012
```

已有同名分支/目录时换新运行ID，不强推。私有包与SHA侧车可用于分析核验，不含秘密或服务令牌。CPU/GPU汇总都不能变成完整ROI识别效果或人工真值；S/B、事件、轨迹/方向仍unknown，unknown不等于安全。Mac落盘需用户另行确认。
