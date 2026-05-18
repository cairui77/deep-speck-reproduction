# Speck32/64 深度学习攻击复现实验

本项目基于公开仓库 `agohr/deep_speck`，复现论文 **Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning** 中的 Speck32/64 神经区分器和 key-rank 相关实验。

## 本次扩展内容

- 增加 `speck_reproduction_1000_cn.py`：1000 行以上中文注释版复现实验脚本。
- 支持 N5/N6/N7/N8 神经区分器评估。
- 支持 ordinary real-vs-random 与 real differences 两类实验。
- 支持解析已有日志并与论文 Table 2、Table 4 对比。
- 支持统计 9 轮 key-rank 补充数据。
- 支持导出 Markdown、CSV、JSON 报告。
- `docs/` 目录中包含按模板填写的开源软件总结文档。

## 快速运行

```powershell
cd D:\codexAI\deep_speck-master
.\.conda-exp3\python.exe .\speck_reproduction_1000_cn.py compare --log .\n5_n8_eval_1e6_paper.log
.\.conda-exp3\python.exe .\speck_reproduction_1000_cn.py keyrank-summary
```

论文级 N5-N8 复现实验：

```powershell
.\.conda-exp3\python.exe .\speck_reproduction_1000_cn.py eval --samples 1000000 --quiet
```

## 说明

该项目用于课程学习、论文复现和实验讲解。原始算法、模型和数据归原论文及原仓库作者所有。
