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

## 安装与启动（Windows / PyCharm）

在 PyCharm 打开 `D:\下载软件\Byzantine`，选择项目的 `.venv` 解释器，然后在终端运行：

```powershell
python -m pip install --upgrade pip
pip install -e ".[local,dev]"
byzantine-app
```

浏览器打开页面后：先进入“设置 → 批量导入”，批量导入 PDF、DOCX、TXT 或 Markdown 文献；再回到“Agent 问答”，先选资料库类型，再选具体书名或全部文献，最后开始新聊天。

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
$env:BYZANTINE_EMBEDDING_MODEL = "D:\下载软件\Byzantine\models\bge-m3"
byzantine-app
```

数据目录包含 `library.db`、`documents/` 和 `qdrant/`，不应提交到 Git；同样不要提交受版权保护的原始 PDF、全文块、模型或 `.env`。

## 验证

```powershell
python -m pytest -q
ruff check src tests
```

旧的单书 CLI 原型仍可供调试：`byzantine-ingest`、`byzantine-chunk`、`byzantine-enrich`、`byzantine-index`、`byzantine-search` 和 `byzantine-ask`。
