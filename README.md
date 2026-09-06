# Historia

Historia 是一个本地运行的拜占庭史研究 Agent。项目目录、Python 包名和 Git 仓库仍保留为 `Byzantine`，界面产品名称统一为 `Historia`。它不把书交给模型“记住”，而是将每段可检索文本保存为带书目、章节、页码和原文区域的 `Evidence`；回答、专题卡和对读记录都附带可回查的出处。

## 当前研究工作台

| 页面 | 用途 |
| --- | --- |
| Agent 问答 | 两层资料范围（个人 / 内置 → 具体文献或全部），多轮上下文聊天；每次回答只检索已选择的文献。可把聊天归档到研究专题。 |
| 研究专题 | 创建研究问题；将对话由 DeepSeek 概括为标题、标签、带出处的研究卡，并保留原始聊天。 |
| 史料平行对读 | 在已选资料范围内选择至少两份文献，按维度并列比较，并保存比较历史。 |
| 矛盾与反证 | 在已选资料范围内检索正向、限制性和替代解释证据，记录可能的冲突或视角差异。 |
| 设置 | 分为“系统与数据”“批量导入”“资料库管理”三个标签页；负责 DeepSeek、索引、路径和文献生命周期。 |

证据阅读器、主张—证据账本、论文证据审计和史料批判卡已从界面移除，避免把 MVP 做成难以使用的功能堆叠。

## 从 GitHub 克隆后运行（Windows / PyCharm）

推荐 Python 3.11–3.13。在 PowerShell 中执行：

```powershell
git clone https://github.com/LifeArtist0102/Byzantine.git
cd Byzantine
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[local,dev]"
```

首次需要下载 BGE-M3 向量模型。模型权重不会被提交到 GitHub；下载后保留在本机即可：

```powershell
pip install modelscope
byzantine-download-model --output-dir models/bge-m3
```

建议把文献、SQLite 和 Qdrant 数据放在 D 盘，而不是项目目录或 C 盘。以下环境变量只对当前 PowerShell 窗口有效：

```powershell
$env:BYZANTINE_DATA_DIR = "D:\ByzantineData"
$env:BYZANTINE_EMBEDDING_MODEL = "$PWD\models\bge-m3"
byzantine-app
```

在 PyCharm 中打开克隆后的 `Byzantine` 文件夹，并把解释器设为 `.venv\Scripts\python.exe` 即可。浏览器打开页面后：先进入“设置 → 批量导入”，每次加入一份带独立书目信息的文献，可连续组成处理队列；再回到“Agent 问答”，先选资料库类型，再选具体书名或全部文献，最后开始新聊天。

如果只想启动界面而暂不进行向量化，可以跳过模型下载；上传后的向量索引会提示模型未准备好。DeepSeek 不是启动界面的必需条件。

## DeepSeek 配置

可以直接在“设置 → 系统与数据”中保存并测试连接，也可以在项目根目录的 `.env` 中填入（不要提交此文件）：

```text
DEEPSEEK_API_KEY=你的密钥
```

未配置密钥时，系统仍可完成上传、向量化、检索和查看证据；但不能生成问答或研究专题摘要。

## 向量检索与本地数据

- PDF 用 PyMuPDF 按页面和文字区域提取；DOCX 保留标题层级、段落和表格结构；文本再按语义块处理。
- BGE-M3 将文本块转为向量，Qdrant 保存“向量 + 文献范围 + 元数据”；SQLite FTS5 同时提供关键词检索，两者用 RRF 融合排序。
- “系统与数据”中的路径复制/打开、DeepSeek 测试/修改、索引检查/重建均连接真实本地服务，不是展示按钮。
- 默认数据目录由系统决定。若要把数据放在 D 盘，可在启动前设置：

```powershell
$env:BYZANTINE_DATA_DIR = "D:\ByzantineData"
$env:BYZANTINE_EMBEDDING_MODEL = "D:\你的项目路径\Byzantine\models\bge-m3"
byzantine-app
```

数据目录包含 `library.db`、`documents/` 和 `qdrant/`，不应提交到 Git；同样不要提交受版权保护的原始 PDF、全文块、模型或 `.env`。

## 验证

```powershell
python -m pytest -q
ruff check src tests
```

旧的单书 CLI 原型仍可供调试：`byzantine-ingest`、`byzantine-chunk`、`byzantine-enrich`、`byzantine-index`、`byzantine-search` 和 `byzantine-ask`。

## 常见问题

- **首次导入大型 PDF 看似停在 30% 左右**：此时程序正在本地用 BGE-M3 分析段落语义并构建 Section—Parent—Child 结构，CPU 会持续工作。资料库的“导入进度”会自动刷新阶段和完成量；请勿重复上传同一文件。
- **关闭页面或重启应用**：待处理文件、处理阶段和已完成文献都保存在 `BYZANTINE_DATA_DIR`。重新打开“设置 → 批量导入”，点击“恢复”或“继续处理”即可；正在处理但尚未落库的那一份会重新处理，已完成部分不会重复导入。
- **`pymupdf` / BGE-M3 / Qdrant 报错**：先确认已执行 `pip install -e ".[local,dev]"`，BGE-M3 目录存在，并且 `BYZANTINE_EMBEDDING_MODEL` 指向该目录。
- **不要提交本地资料**：原始 PDF、数据库、Qdrant 向量、下载的模型以及 `.env` 都被 `.gitignore` 排除；它们可能包含版权内容或密钥。
