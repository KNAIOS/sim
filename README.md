# 中期报告实验与数据包

整理日期：2026-11-09。对应当前论文的 BA-RL4 联合子阵激活与功率分配实验，原项目未移动、未修改。本包不是旧版 DQN/LSTM 实验的改名版本。

## 从 GitHub 获取完整材料

仓库：https://github.com/KNAIOS/sim 。仓库根目录即本地 `simu` 文件夹的内容。

数组数据 `.npz` 和模型权重 `.pt` 使用 Git LFS。请在具备 Git LFS 的环境中克隆并拉取数据：

```powershell
git clone https://github.com/KNAIOS/sim.git
cd sim
git lfs pull
```

如果只有几行以 `version https://git-lfs.github.com/spec/v1` 开头的文字，拿到的是 LFS 指针而不是真实数据，应执行 `git lfs pull`。网页下载 ZIP 不能作为完整数据已到位的保证；下载后运行 `verify` 可检查所有存档文件。不会自动安装 Python 包。

仓库不包含虚拟环境、缓存和 `runs/replotted` 重绘输出；原始论文图在 `figures`，重绘结果可以用后文的 `plot` 命令生成。

## 先看这些文件

1. `中期报告材料说明.md`：研究在做什么、实验如何组织、可以支持哪些结论。
2. `实验与代码索引.md`：实验、数据、训练代码、论文图表的一一对应位置。
3. `数据字段说明.md`：能耗、速率不达标率、误差棒、活动子阵数等指标含义。
4. `figures/`：当前论文两张图的 PNG、PDF 和 SVG 原件，可用于报告。
5. `tables/`：三组主要实验的 CSV 汇总、原始回合记录及配对统计的快捷副本。

## 文件组织

```text
simu/
  README.md
  中期报告材料说明.md
  实验与代码索引.md
  数据字段说明.md
  figures/                    当前论文两张图，三种格式
  tables/                     主要数据快捷副本
  reproduction/
    xl_mimo_sim/
      future_vr/              几何信道、硬件、状态与参数
      joint_power/            联合控制、训练、评价、测试与阶段报告
      major_revision/         当前代码和存档校验所依赖的历史模块
      src/、tests/            原始支持代码与测试
      results/joint*/         原始结果、逐时隙账目、训练样本与权重
    standard_channel_20260920/ 标准化 UMi 信道与历史数值测试样例
    paper_assets/             当前图表的原始绘图代码和表格源码
    ICECAI2026_4pages_20260922/ 旧校验清单依赖的冻结文件，不是当前投稿入口
  tools/                      校验和重绘入口
  run.ps1                     Windows 操作入口
  PACKAGING_MANIFEST.json     每个复制文件的来源、大小和 SHA-256
  runs/                       本包校验和重绘输出，不覆盖存档结果
```

保留 `reproduction` 下的相对目录，是因为原代码依赖这些路径。历史结果没有删除；使用中期报告时请按索引选取当前证据，不能把早期阶段报告中的进度描述当作今天的状态。未复制原有虚拟环境、缓存、远程连接配置或所有早期独立研究的结果。

## 如何使用

仅写报告不需要运行仿真：直接看说明、`tables` 和 `figures` 即可。

所有操作默认使用本包 `.venv/Scripts/python.exe`，也可通过 `-Python` 指定已有解释器。本次没有创建虚拟环境、安装依赖或启动完整训练。原标准信道环境记录在 `reproduction/standard_channel_20260920/requirements.lock.txt`，历史 Python 为 3.12.13，PyTorch 为 2.9.1+cpu，Sionna 为 2.1.0；这些是原环境记录，不表示已在任意新机器上验证安装。

```powershell
# 在 simu 目录运行；把路径替换为你选用的解释器。
.\run.ps1 -Action verify -Python 'C:\path\to\python.exe'
.\run.ps1 -Action plot   -Python 'C:\path\to\python.exe'
.\run.ps1 -Action tests  -Python 'C:\path\to\python.exe'
```

- `verify` 只用 Python 标准库，核对复制文件、冻结证据、120 个回合及能耗汇总，结果写入 `runs/verification.json`。
- `plot` 需要 NumPy 和 Matplotlib；从已有 CSV 重新生成两张图与两张 LaTeX 表，写入 `runs/replotted/`。不会修改 `figures/` 或原始 CSV。
- `tests` 需要 NumPy、PyTorch 及原测试依赖；运行联合控制测试，不是重新完成全部实验。
- 包内脚本只临时设置当前进程的缓存与线程参数，退出时恢复，不更改系统或用户级环境变量。

## 关于训练与完整重跑

训练样本、验证样本、训练程序、已选权重和训练历史均已保留，具体入口见索引。原始研究脚本使用固定输出目录和 `exist_ok=False`，直接运行时会保护现有存档而拒绝覆盖。这是有意保留的行为。

若要重训或完整重跑，应另建新的运行目录并配置输出路径；不要删除或改名本包 `results` 来绕过保护。部分原始历史清单包含旧绝对路径，`tools/verify_bundle.py` 会映射到本包再校验；原始清单没有被改写。当前提供的是可独立查看、核验、重绘和测试的实验材料包，不声称所有历史流水线已变成一键重训程序。

本次包装核验见 `PACKAGE_CHECK.md`。后续写中期报告可直接以本包说明作为 `ieee-writing` 的输入；若需补实验，再使用 `ieee-experiment` 明确问题和验证范围。
