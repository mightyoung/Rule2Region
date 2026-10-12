# 固定提示词 × 三机位边界候选对照交接

状态：**最小任务说明已交接，完整可运行包仍待补。** input_manifest.json、model_manifest.json、experiment.json 已冻结本轮输入和参数。forward.py 是已有真实前向脚本的 device/prompt 参数化版本，尚未完成本轮测试；暂勿将此初版当成一键可运行实验。待补：安全原图提取、外层真实运行记录、结果验证器、输出 schema、CPU 离线检查和回传命令。

数据足以验证候选对提示词的敏感性、截边行为和软件可执行性，**不足以验证完整 ROI 准确率、S/B 或进出事件**。原 C1 图 threshold_not_visible 是限定的人工可见性结论，左右贴合仅在另一派生图被确认；C2/C3 尚待人工复核。不能重造真值。当前三机位近侧开口下界均截断。远侧关门框不是可通行开口真值。

仅用已授权 [ChokePoint 原发布](https://zenodo.org/records/815657) P1E_S1.tar.xz（三机位各原帧0），精确文件与哈希见 input_manifest.json。许可 CC BY NC 4.0，遵守[官方附加条件](https://arma.sourceforge.net/chokepoint/)：非商业、NICTA及Wong等2011论文署名、保留原许可、标明衍生材料。原图/标注/张量/可视化只私有回传，禁止提交公开 GitHub；不包含第二批83框或HSV材料。

模型为既有 GroundingDINO tiny revision a2bb814dd30d776dcf7e30523b00659f4f141c71，9文件哈希见 model_manifest.json。固定 combined / door_only / doorway_only；3原图×3提示，最多9次前向；.25/.25、same-exact-label NMS IoU>.7、CPU或CUDA float32、batch1、seed0，无训练/调阈值。GPU是可选加速，CPU也能运行，不能称必须GPU。运行范围不含算力启动、付费调用或密钥配置。

遇许可/哈希不匹配、OOM、非有限张量、任一运行失败，停止并保留unknown与失败日志；不自动换模型/图像/精度/参数重试。缺底界证据时停止完整ROI解释。unknown不等于安全。始终保留轨迹与方向unknown，方向变化只是证据，模糊目标才调用复合研判。既有冻结协议不修改。
