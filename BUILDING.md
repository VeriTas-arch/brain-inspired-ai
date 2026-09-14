# 课程材料的组织与导出

课程源码集中维护在一个仓库中，每个主题可以单独导出为课程包。下载解压后，在 VS Code 中打开 Notebook 即可学习；想调整内容或添加课程，可以从下面几个文件入手。

## 文件如何组织

| 位置 | 用途 |
| --- | --- |
| [biai/](biai/) | 按主题存放课程，每个目录包含 README、Notebook 和所需的 Python 代码 |
| [biai/paths.py](biai/paths.py) | 定义数据、结果和参考素材的位置 |
| [biai/reproducibility.py](biai/reproducibility.py) | 提供随机种子等共用设置 |
| [courses.toml](courses.toml) | 列出各课的标题、需要导出的文件和依赖名称 |
| [pyproject.toml](pyproject.toml) | 维护整个课程仓库的发布版本和依赖的版本要求 |
| [scripts/export_course.py](scripts/export_course.py) | 生成课程 ZIP 和对应的展开目录 |
| [tests/](tests/) | 检查课程代码、文档链接和导出后的运行情况 |

例如，入门课的源文件是 `biai/intro/README.md` 和 `biai/intro/intro.ipynb`。Atari 的代码较多，算法、环境和训练循环放在 `biai/atari/` 的 Python 文件中，Notebook 负责讲解与调用。

修改前先阅读[仓库维护规则](AGENTS.md)。修改 Atari 实现时，同时遵循
[Atari 实现约定](biai/atari/AGENTS.md)；测试的组织与运行约定见
[测试维护规则](tests/AGENTS.md)。这些维护文档不随学生课程包分发。

## 导出一节课

先按 [README 的完整仓库安装说明](README.md#使用完整仓库)准备环境，然后在仓库根目录执行：

```bash
python scripts/export_course.py intro
```

所有课程统一使用根目录 `pyproject.toml` 的 `[project].version`。以 `0.1.0` 为例，输出结构如下：

```bash
dist/intro/
├── biai-intro-0.1.0.zip
└── biai-intro-0.1.0/
    ├── README.md
    ├── intro.ipynb
    ├── pyproject.toml
    ├── LICENSE
    └── biai/
```

展开目录与 ZIP 中的文件一致，可以直接打开检查。导出脚本会把本课的 README 和 Notebook 放到根目录，调整本地链接，再带上所需的公共代码。README 中的依赖安装命令会按 `pyproject.toml` 补齐版本要求。

把命令中的 `intro` 换成 `mlp`、`cnn`、`continual_mnist`、`meta_learning` 或 `atari`，即可导出对应主题。Atari 包还会包含参考图表与动图；数据集、本地训练结果和模型不随课程包分发。

发布新版本前，更新根目录 `pyproject.toml` 的 `[project].version`，再导出所需课程。ZIP 文件名、内部根目录、展开目录、包内项目版本和 ZIP 注释中的版本保持一致。已有同名 ZIP 或目录时，脚本会停止，避免覆盖之前的内容。使用 `--output-dir` 可以更换输出根目录。

导出读取当前目录中的文件，包括尚未提交的修改。ZIP 注释中记录发布版本、来源提交和修改状态，便于之后核对。正式发布整套课程时，从同一个已检查的源码状态生成各课包。

## 添加新课程

在 `biai/` 下创建主题目录，放入 `README.md`、与目录同名的 `.ipynb` 文件和 `__init__.py`。README 可以参考已有课程，保留依赖安装命令两侧的 `course-dependencies` 注释，供导出脚本更新版本要求。

在 `courses.toml` 中增加该主题，列出所需文件和依赖名称。新增依赖的版本要求写在 `pyproject.toml` 中，课程目录与标题不必带课次编号。

最后更新根 README 的课程目录，并为新课程补充短运行检查。现有导出检查见 [test_course_exports.py](tests/test_course_exports.py)，它会核对 ZIP 与展开目录的内容，并在独立进程中运行课程示例。
