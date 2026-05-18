# Speck32/64 深度学习攻击复现实验

本仓库用于复现论文 **Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning** 中的 Speck32/64 神经区分器与 key-rank 实验。

## 主要内容

- `eval.py`：N5/N6/N7/N8 预训练神经区分器评估。
- `key_rank.py`：9 轮 key-rank 攻击质量统计。
- `docs/`：按模板填写的开源软件总结文档。

## 快速运行

```powershell
cd D:\codexAI\deep_speck-master
.\.conda-exp3\python.exe .\speck_reproduction_1000_cn.py compare --log .\n5_n8_eval_1e6_paper.log
.\.conda-exp3\python.exe .\speck_reproduction_1000_cn.py keyrank-summary
```

论文级 N5-N8 实验：

```powershell
.\.conda-exp3\python.exe .\speck_reproduction_1000_cn.py eval --samples 1000000 --quiet
```
## 项目声明 Project Statement

**项目信息：**

- **项目名称 (Project Name)**：deep-speck-reproduction
- **项目作者 (Author)**：cairui, chengzheng
- **作者单位 (Affiliation)**：暨南大学网络空间安全学院 (College of Cyber Security, Jinan University)

> 💡 **说明**：本项目仅用于课程学习、论文复现和实验讲解。原始算法、模型和数据归原论文及原仓库作者所有。
## 说明

本项目仅用于课程学习、论文复现和实验讲解。原始算法、模型和数据归原论文及原仓库作者所有。
