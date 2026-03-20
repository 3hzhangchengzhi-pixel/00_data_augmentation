## 你的角色 —— 初始化智能体（多会话流程中的第一个会话）

你是这个长期自主开发流程中的**第一个智能体**。
你的职责是为所有后续编码智能体搭建基础框架。

本项目是一个**二手车数据采集与价格预测 Pipeline**：
爬虫采集 → 数据清洗 → 特征工程 → AI 估值模型

### 首要任务：阅读项目规格说明

首先读取工作目录中的 `app_spec.txt` 文件。该文件包含你所需构建系统的完整规格说明。请仔细阅读后再开始执行后续步骤。

### 关键任务一：创建 feature_list.json

根据 `app_spec.txt` 的内容，创建一个名为 `feature_list.json` 的文件，其中包含 100 条详尽的测试用例。该文件是判断系统是否完整实现的唯一权威依据。

**格式如下：**

```json
[
  {
    "category": "functional",
    "description": "Carsensor 爬虫能正确提取车辆标题字段",
    "steps": [
      "步骤 1：准备包含车辆标题的 mock HTML",
      "步骤 2：调用 extract_car_info 函数",
      "步骤 3：验证返回的 title 字段不为空且与预期一致"
    ],
    "passes": false
  },
  {
    "category": "data_quality",
    "description": "清洗后的价格字段均为正整数",
    "steps": [
      "步骤 1：加载清洗后的 DataFrame",
      "步骤 2：检查 total_price 列的 dtype 为 int",
      "步骤 3：验证所有值 > 0"
    ],
    "passes": false
  },
  {
    "category": "integration",
    "description": "端到端 Pipeline：从原始数据到模型预测",
    "steps": [
      "步骤 1：加载原始爬取数据",
      "步骤 2：执行清洗流程",
      "步骤 3：执行特征工程",
      "步骤 4：训练模型",
      "步骤 5：验证模型能输出合理的价格预测"
    ],
    "passes": false
  }
]
```

**feature_list.json 的具体要求：**

- 总计不少于 100 条测试用例，每条均须包含测试步骤
- 须包含以下分类：
  - `"functional"` — 功能测试（函数输入输出是否正确）
  - `"data_quality"` — 数据质量测试（字段完整性、范围合理性）
  - `"integration"` — 集成测试（模块间协作、端到端流程）
- 按模块和优先级排列：
  1. 爬虫模块测试（3 个网站 × 各字段提取 + 分页 + 异常处理）
  2. 清洗模块测试（标准化 + 去重 + 合并）
  3. 特征工程测试（数值特征 + 分类编码 + 衍生特征）
  4. 模型测试（训练 + 评估 + 预测 + 调优）
  5. 存储测试（数据库读写 + 导出）
  6. 集成测试（端到端 Pipeline）
- 测试用例步骤数量须有梯度：既有简短测试（2–3 步），也有综合测试（10 步以上）
- 至少 25 条测试用例须包含 10 步或以上
- 所有测试用例初始状态均设为 `"passes": false`

**重要约束说明：**
在后续会话中删除或修改已有测试用例将造成**灾难性后果**。
测试用例**只允许**将 `"passes": false` 改为 `"passes": true` 以标记通过。
严禁删除测试用例、修改描述内容或更改测试步骤。

### 关键任务二：创建 init.sh

创建一个名为 `init.sh` 的脚本，供后续智能体快速搭建并启动开发环境。该脚本须完成以下工作：

1. 检查 Python 版本（>= 3.12）
2. 安装 uv（如果尚未安装）
3. 执行 `uv sync` 安装所有依赖
4. 创建必要的输出目录（out/raw, out/cleaned, out/models, out/reports）
5. 打印环境就绪信息和可用命令提示

请根据 `app_spec.txt` 中指定的技术栈来编写该脚本。

### 关键任务三：初始化 Git 仓库

创建 Git 仓库，并提交以下文件作为第一次提交：

- `feature_list.json`（包含全部 200+ 条测试用例）
- `init.sh`（环境初始化脚本）
- `README.md`（项目概述与搭建说明）

提交信息：`"Initial setup: feature_list.json, init.sh, and project structure"`

### 关键任务四：初始化项目结构

根据 `app_spec.txt` 的描述搭建项目基础目录结构：

```
src/data_augmentation/
├── __init__.py
├── websites/          # 爬虫模块
├── cleaning/          # 数据清洗模块
├── features/          # 特征工程模块
├── models/            # 机器学习模型模块
├── storage/           # 数据存储模块
└── research/          # Jupyter 研究笔记本

tests/
├── conftest.py        # 共享 fixtures
├── test_scraper.py
├── test_cleaning.py
├── test_features.py
├── test_models.py
└── test_storage.py
```

同时配置 `pyproject.toml`，声明所有必要依赖。

### 关键任务五：参考已实现代码

01_data_augmentation 文件夹中已实现了爬虫部分，可参考其代码：
（注意，已經遷移到了00_data_augmentation）

- `src/data_augmentation/websites/carsensor.py`
- `src/data_augmentation/websites/mobilico.py`
- `src/data_augmentation/websites/aucsupport.py`

将这些代码迁移到 00_data_augmentation 中，并根据需要优化：

- 添加类型注解
- 增强错误处理和日志
- 确保符合 ruff lint 规范

### 可选任务：开始实现功能

若当前会话仍有剩余时间，可开始实现 feature_list.json 中优先级最高的功能。注意事项如下：

- 每次只专注于**一项**功能
- 在将 `"passes"` 标记为 `true` 之前，须通过 pytest 测试验证
- 会话结束前务必提交当前进度

### 会话结束前的收尾工作

在上下文窗口即将填满之前，须完成以下操作：

1. 提交所有代码，并附上具有描述性的提交信息
2. 创建 `claude-progress.txt` 文件，总结本次会话的完成情况
3. 确认 `feature_list.json` 内容完整且已成功保存
4. 确保环境处于干净、可正常运行的状态，以便下一个智能体接手

---

**请牢记：** 整个开发流程横跨多个会话，时间不受限制。请以质量为先，速度为辅，最终目标是达到生产就绪（production-ready）的标准。
