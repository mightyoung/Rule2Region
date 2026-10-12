# 固定提示词 × 三机位边界候选对照交接

状态：**可执行交接包已补齐，CPU离线预检与结果验证通过；CUDA路径尚待用户运行验证。** 从 [RUNBOOK.md](RUNBOOK.md) 开始，输入/模型/参数清单、合法原归档导入、真实前向执行器、失败unknown保留、私有输出schema和严格结果验证器、资源预算与回传命令均已提供。[VALIDATION.json](VALIDATION.json) 如实记录10项合成契约测试、2次真实CPU smoke（首轮验证器误拒文本padding的正常-inf，修正后第二轮通过）、产物篡改拒绝检查。合成样例不是研究效果或人工真值，未运行完整9组CUDA矩阵。GPU显存预算为估计，CUDA轮子/驱动实测与全传递依赖wheel哈希锁尚未验证，详细限制见RUNBOOK。

数据足以验证候选对提示词的敏感性、截边行为和软件可执行性，**不足以验证完整 ROI 准确率、S/B 或进出事件**。原 C1 图 threshold_not_visible 是限定的人工可见性结论，左右贴合仅在另一派生图被确认；C2/C3 尚待人工复核。不能重造真值。当前三机位近侧开口下界均截断。远侧关门框不是可通行开口真值。

仅用已授权 [ChokePoint 原发布](https://zenodo.org/records/815657) P1E_S1.tar.xz（三机位各原帧0），精确文件与哈希见 input_manifest.json。许可 CC BY NC 4.0，遵守[官方附加条件](https://arma.sourceforge.net/chokepoint/)：非商业、NICTA及Wong等2011论文署名、保留原许可、标明衍生材料。原图/标注/张量/可视化只私有回传，禁止提交公开 GitHub；不包含第二批83框或HSV材料。

模型为既有 GroundingDINO tiny revision a2bb814dd30d776dcf7e30523b00659f4f141c71，9文件哈希见 model_manifest.json。固定 combined / door_only / doorway_only；3原图×3提示，最多9次前向；.25/.25、same-exact-label NMS IoU>.7、CPU或CUDA float32、batch1、seed0，无训练/调阈值。GPU是可选加速，CPU也能运行，不能称必须GPU。运行范围不含算力启动、付费调用或密钥配置。

遇许可/哈希不匹配、OOM、非有限张量、任一运行失败，停止并保留unknown与失败日志；不自动换模型/图像/精度/参数重试。缺底界证据时停止完整ROI解释。unknown不等于安全。始终保留轨迹与方向unknown，方向变化只是证据，模糊目标才调用复合研判。既有冻结协议不修改。
