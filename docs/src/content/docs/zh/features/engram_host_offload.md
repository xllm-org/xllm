---
title: "Engram 主机侧卸载"
sidebar:
  order: 83
---

## 概述

DeepSeek V4.1 W8A8 权重可以将 int8 Engram 表放在主机内存中，减少 HBM 占用。节省量取决于权重及 TP 分片。各 rank 独立计算 hash ID，通过 `xllm_ops.engram_int8_lookup_cpu` 查找并去量化所需行，再上传到设备。

卸载会增加主机查表和数据传输开销。ID 下载批量提交后统一等待当前 stream；复用 pinned staging 内存前，会等待上一次上传完成。每个 graph entry 独立持有设备行缓冲区，eager batch 扩容不会替换已捕获的地址。

## 配置

通过 `engram_config` gflag 传入 JSON 对象：

```bash
xllm --model /path/to/model --engram_config '{"cpu_offload":true}'
```

| 键 | 默认值 | 含义 |
|---|---|---|
| `cpu_offload` | `false` | 将 int8 Engram 表放在主机内存中。 |
| `dp_shared_memory` | `false` | 节点内 DP 副本共享只读 TP 分片，要求 `cpu_offload=true`。 |

不传参数或传入 `{}` 时使用 HBM 路径。开启主机表共享：

```bash
xllm --model /path/to/model --engram_config '{"cpu_offload":true,"dp_shared_memory":true}'
```

各 DP rank 的输入和历史不同，仍分别执行查表与上传。共享对象是表数据，不是查表结果。

字段必须为 JSON 布尔值。未知字段、重复字段、非法 JSON，以及未开启卸载时设置 `dp_shared_memory=true`，都会报配置错误。

## 约束

- 主机卸载目前支持 TP=1、2、4、8，要求 PP=1、DCP=1。这些卸载专用检查不改变原有 HBM 路径的拓扑约束。
- TP 分片按向上取整的列数分配，多余列补零并在 gather 后裁掉；hash 列数不必被 TP 整除。
- 卸载支持 safetensors 或 dummy 权重，仅 int8 表具有 CPU 查表路径。其他存储格式保留 HBM 路径，不能请求主机表共享。
- 主机表共享使用节点内 Gloo 通信组，需要足够的 `/dev/shm` 容量及相同的共享内存 namespace。attach 后会核对段标识；hostname 相同不能单独证明共享成功。
- 表共享减少主机内存占用，不减少各 rank 的查表和传输工作。

## 实现位置

- 配置：`xllm/python/models/engram_config.py`
- Hash、staging 与执行 metadata：`xllm/python/models/deepseek_v41_engram.py`
- 主机共享存储：`xllm/python/models/engram_shared_memory.py`
- CPU 查表：`xllm/core/kernels/npu/xllm_ops/engram_int8_lookup_cpu.cpp`
