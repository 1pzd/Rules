# Rules Pipeline

本项目包含两个主要脚本：`rule_generator.py` 和 `rule_checker.py`。

## 代码协作关系

`rule_generator.py` 负责生成规则，是主流程脚本。它会从本地数据集中抽取正常样本图片，调用模型生成每个类别的正常外观规则，并把规则保存到 `generated_rules.json` 和 `Rules.txt`。

`rule_checker.py` 负责检查图片，是底层检查器。它接收一条规则和一张图片，调用模型判断图片是否符合规则，并输出 `y` 或 `n`。

两者的关系是单向的：

```text
rule_generator.py -> 调用 rule_checker.check_image()
```

也就是说，`rule_generator.py` 在生成规则后，会复用 `rule_checker.py` 的检查能力，对生成的规则进行验证。`rule_checker.py` 本身不依赖 `rule_generator.py`，也可以单独运行。

## 整体输入

主要输入包括：

- 本地数据集目录：
  - `mvtec_anomaly_detection/`
  - `mvtec_loco_anomaly_detection/`
  - `VisA_20220922/VisA_20220922/`
- 正常样本图片：用于生成规则
- 测试图片：用于验证规则
- 规则文件：默认使用 `Rules.txt`
- 待检查图片：用于单独执行规则检查
- 模型 API 配置：定义在 `rule_checker.py` 中，并由 `rule_generator.py` 共享

## 整体输出

`rule_generator.py` 的输出：

- `generated_rules.json`：结构化规则结果
- `Rules.txt`：按数据集和类别整理的可读规则文件
- 控制台日志：显示生成、验证和精炼过程

`rule_checker.py` 的输出：

- `y`：图片符合指定规则
- `n`：图片不符合指定规则
- 错误信息：例如规则文件缺失、未指定 `--rule-key`、API 调用失败等

## 常用命令

生成全部规则：

```powershell
python rule_generator.py
```

只生成单个类别规则：

```powershell
python rule_generator.py --single mvtec_ad/bottle
```

使用 `Rules.txt` 检查单张图片：

```powershell
python rule_checker.py path\to\image.png --rule-key mvtec_ad/bottle
```

使用单独规则文件检查单张图片：

```powershell
python rule_checker.py path\to\image.png --rules-file single_rule.txt
```

## 简要流程

```text
数据集图片
  -> rule_generator.py 生成规则
  -> 输出 Rules.txt / generated_rules.json
  -> rule_checker.py 使用规则检查图片
  -> 输出 y 或 n
```
