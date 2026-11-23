# 统一内存 Agent workload 仿真

用 SWE-bench/OpenHands 多轮轨迹重放单台统一内存设备上的本地推理与工具执行。比较固定分区、同预算共享池、两个同预算 oracle 和宽预算诊断组，衡量内存排队、KV 重算、工具恢复与端到端表现。

## 运行

需要 Python 3.10 或更新版本和本地历史 trace。仿真使用标准库，可在普通主机运行。参数入口为 [experiment_config.py](experiment_config.py)，其中的 `TRACE_ROOT` 默认指向 `~/codes/experiment-jetson`。
运行单元测试
```bash
python3 -m unittest discover -s tests -v
```
运行仿真
```bash
sim_run_dir="results/sim-$(date +%Y%m%d-%H%M%S)"
python -m sim run --config experiment_config.py --out "$sim_run_dir" --quick
```

每次运行指定一个新目录。加上 `--quick` 时，使用第一个 seed 运行 `concurrency` 中所有并发度的 `burst` 场景，跳过 `staggered` 和敏感性分析。每进入一个场景会打印进度，`status.json` 记录执行状态。

## 数据与内存账本

- 导入同批 Xarray、scikit-learn、Astropy 的 `per_call_tokens.csv`、`tool_calls.csv`、`resources.csv` 和 `summary.json`。推理回合按响应 ID 关联工具调用，一轮中重叠的工具区间按时长并集重放。常驻量取首末工具窗口之间的有效空闲样本中位数，每条轨迹的执行期统一使用其工具窗口内的采样峰值。工具开销包含 OpenHands 轨迹中的调用到返回时间。
- 推理以历史请求输入和输出 token 数估计本地请求规模，用历史缓存读取量近似同一 Agent 的前缀。新 token 按配置中的 prefill 和 decode 批吞吐处理。每次准入预留当轮峰值 KV，推理推进时把预留量转为物理驻留，下一轮只复用该 Agent 自身的前缀。恢复时只重算丢失的可复用前缀，恢复在批边界独占 GPU。
- 工具保留环境常驻量，工作期间占用执行峰值。空闲环境经过保存、写入与释放阶段后才交还 DRAM。恢复预留执行与恢复峰值中的较大值，读盘和恢复可以与其他 agent 的执行重叠。磁盘读写使用同一个 FIFO 通道。历史恢复总时长用于校准，读盘份额是配置代理。
- 固定分区在扣除静态保留后限制每侧额度，两侧各用自己的 FIFO 和本侧 LRU。共享池把推理与工具申请放进统一 FIFO。队首容量不足时，全局 LRU 可以驱逐任意空闲 KV 或工具环境，实际释放后再准入。宽预算组使用相同的准入次序。
- 共享池和宽预算组都允许驻留命中的请求直接准入。所需 KV 前缀或工具环境仍在内存时，只要本轮新增用量能放入未预留的额度，请求就可以提前通过。冷请求、需要恢复的请求和无法直接容纳的请求按 FIFO 等待。实际申请超限和队列无法继续推进分别记账。

## 同预算 Oracle

- `oracle_fifo` 沿用共享池的 FIFO 和驻留命中直通，在队首遇到内存阻塞时比较等待自然释放与跨侧驱逐。`oracle` 同时比较让其他已提交请求先准入的收益，包括冷请求和需要恢复的请求。参与重排的候选取 FIFO 前八个，以及共享池本来会直通的请求。
- 每个回收候选从两侧空闲状态里选出足以覆盖缺口的一组。选取顺序分别使用 LRU、预计释放时间、单位释放字节的释放与恢复代价。队尾请求发起回收时，释放的额度保留给该请求，直到事务完成。
- 每种动作从当前状态独立预演一次，随后按共享池规则继续。窗口默认 30 秒，可在实验配置中设置 `oracle_window_s`。预演可读取窗口内的未来到达和轨迹，遵守 DRAM、GPU 和磁盘账本。评分先比较预测失败数，再比较累计未完成时间加剩余服务时间估计。后者包括窗口末的 KV 重算和工具恢复负担。每次只执行代价最低的动作，同分优先选择共享池的动作。
- `shared → oracle_fifo` 衡量等待与驱逐的收益，`oracle_fifo → oracle` 衡量加入准入调度的净收益。结果给出当前模型下已实现的同预算改善幅度，全局最优改善幅度仍未知。宽预算组提供容量放宽后的诊断参考。

## 结果

| 文件 | 内容 |
| --- | --- |
| `config.json`、`sources.json` | 生效参数、场景配置、配置与 trace 源文件的 SHA-256 |
| `traces.json`、`workloads.json` | 归一化 trace 与各对照组共用的到达清单 |
| `runs.csv`、`agents.csv` | 逐 seed 的运行和 Agent 指标、失败原因、队首和队尾等待、驻留命中直通和 oracle 越序准入 |
| `events.jsonl.gz` | 纳秒粒度的事件、内存账本、跨侧驱逐，以及 oracle 候选、预测代价和最终选择 |
| `comparisons.csv`、`report.md` | 最佳固定分区、累计配对延迟节省、尾延迟、吞吐和开销归因 |

Goodput 的分母为首个到达到最后一个任务终态之间的时间。部分失败时结合完成数和成对完成样本一起解读。结果是当前假设下的仿真证据。真实设备的内存占用、批吞吐和恢复开销仍需独立校准。

`results/v1` 的历史共享池采用两侧分别排队、同侧回收。`results/v2` 对应统一 FIFO、驻留命中准入和全局 LRU。

## 内存利用率可视化

使用项目专用环境和独立的绘图依赖。下面两组用同一份 16 GiB、8 并发、seed 0 的工作负载做对照。
安装专用环境
```bash
python -m pip install -r requirements-viz.txt
```
运行绘制脚本
```bash
conda activate aws-py310
python -m sim plot-memory \
  --run "$sim_run_dir" \
  --case burst/n8/seed0/fixed90 \
  --case burst/n8/seed0/shared \
  --out "${sim_run_dir}-memory"

python -m sim plot-memory \
  --run "results/sim-20261123-120246" \
  --case burst/n8/seed0/fixed90 \
  --case burst/n8/seed0/shared \
  --case burst/n8/seed0/relaxed \
  --case burst/n6/seed0/fixed90 \
  --case burst/n6/seed0/shared \
  --case burst/n6/seed0/relaxed \
  --case burst/n3/seed0/fixed90 \
  --case burst/n3/seed0/shared \
  --case burst/n3/seed0/relaxed \
  --out "results/sim-20261123-120246-memory"
```

图中使用仿真分钟和 GiB。**物理驻留**为推理 KV 与工具当前占用的字节数，包含空闲或等待释放的状态。**正在执行**为当前 prefill、decode、reprefill 批次的驻留 KV，加上执行调用、读取 checkpoint 或恢复中的工具内存。磁盘排队期间，恢复内存只计入物理驻留。两条曲线的差值表示未参与当前执行的驻留量。

每个点按各状态在该分钟内持续的仿真时间计算加权平均，最后不足一分钟按实际时长计算。水平虚线表示总 DRAM 扣除模型和系统静态保留量后的预算。每个面板标题用同一预算作分母，分别给出全程时间加权的平均驻留占用（`mean resident / budget`）和平均执行占用（`mean active / budget`）。

`memory_minute.csv` 保存图中的区间和均值，以及以 0–1 表示的每分钟执行占用与驻留占用之比。`memory_timeline.csv` 保存计算前的事件时点与内存快照。CSV 使用纳秒和字节。

`--case` 可多次指定，case ID 见运行目录的 `runs.csv`。对旧运行（如 `results/v1`）加 `--resident-only` 可生成物理驻留曲线，其执行数据列为空。每次输出到新目录。
