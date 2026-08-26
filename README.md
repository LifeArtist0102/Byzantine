# Byzantine

一个在本机运行、单用户使用的拜占庭历史研究 Agent MVP。它不是把书丢给模型“记住”，而是将每一段可检索文本保存为带有文献、版本、页码和原文区域的统一 `Evidence`；Agent、对读、主张账本、论文审计与反证检索都复用这份证据。

## 已实现的本地闭环

- 自动创建 `starter`（基础）与 `personal`（个人）两个资料库，并以 SQLite 持久化文献、全文检索、专题、主张、对读、审计和史料批判卡。
- 导入 PDF、TXT、Markdown、JPG、JPEG、PNG；PDF 用 PyMuPDF 保留文字块与坐标，图片／扫描件在安装可选 OCR 后用 PaddleOCR 处理。
- 保存原始文件、文件哈希、书目信息、段落／页码、前后 chunk、原文区域和元数据；重复文件会被拒绝。
- SQLite FTS5 关键词检索与 BGE-M3 + 本地 Qdrant 向量检索用 RRF 融合。若可选向量依赖未安装，文本全文检索仍可用，并会显示原因。
- DeepSeek 只接收当次检索得到的证据；输出必须使用 `[S1]` 等合法编号。书名、版本、页码和两种脚注由程序生成，不能由模型编造。
- Streamlit 页面提供问答、上传、资料库、专题、平行对读、主张—证据账本、论文证据审计、矛盾与反证、史料批判卡和设置。

## Windows 安装与启动

在 PyCharm 打开 `D:\下载软件\Byzantine`，使用 Python 3.11–3.13 创建 `.venv`，然后在终端运行：

```powershell
python -m pip install --upgrade pip
pip install -e ".[local,dev]"
```

首次使用 Streamlit：

```powershell
byzantine-app
```

浏览器打开后，先进入“上传资料”，选择 `personal` 或 `starter`，填写书名、作者、版本、语言和资料类型，再点击“确认书目信息并处理”。处理完成后可在“Agent 问答”检索。

若安装 OCR（图片与扫描 PDF 才需要）：

```powershell
pip install -e ".[ocr]"
```

模型权重默认由 BGE-M3 首次使用时下载；如已将模型下载到本地，可设置：

```powershell
$env:BYZANTINE_EMBEDDING_MODEL = "D:\你的模型目录\bge-m3"
```

## DeepSeek 配置

复制 `.env.example` 为 `.env`，只填写本机私有密钥：

```text
DEEPSEEK_API_KEY=你的密钥
```

`.env`、模型、资料、SQLite 数据库、Qdrant 索引和 OCR 结果均被 Git 忽略。未配置密钥时，界面仍可检索和检查证据，但不会生成模型回答。

## 数据位置与迁移

默认数据目录由 `platformdirs` 决定，通常位于 Windows 用户应用数据目录，包含：

```text
APP_DATA/
├── library.db
├── documents/<document_id>/source.*
└── qdrant/
```

开发／测试时可指定一个独立目录：

```powershell
$env:BYZANTINE_DATA_DIR = "D:\ByzantineData"
byzantine-app
```

不要将受版权保护的 PDF、全文 chunks、模型或该目录提交到 Git。`starter` 机制只提供资料库容器；项目所有者可在本地导入其合法副本。

## 保留的命令行原型

既有单本书命令仍可使用：

```powershell
byzantine-ingest "D:\C盘迁移\Desktop\The Oxford Handbook of Byzantine Studies.pdf"
byzantine-chunk
byzantine-enrich
byzantine-index
byzantine-search "Why did Basil II strengthen imperial authority?"
byzantine-ask "Why did Basil II strengthen imperial authority?"
```

新的多文献工作流不再依赖 `config/book.yaml` 的单一书目设置；该配置仅为兼容上述 CLI 原型而保留。

## 研究功能如何使用

- **平行对读**：先导入至少两本文献，再输入问题和维度；每个文献—维度格保留自己的 `Evidence`，不把不同材料混成一段结论。
- **主张账本**：输入一条可检验主张，系统保存初始背景证据；由研究者标记支持、反对、限制或背景，系统不会按“支持数量”判定真伪。
- **论文审计**：粘贴草稿，系统逐句检索并标示 `supported`、`unsupported` 或 `overstated`，不会覆盖原稿。
- **矛盾与反证**：对同一问题执行原问题、反面／限制、不同解释三类查询；差异会谨慎区分为潜在直接冲突、翻译差异或不同视角。
- **史料批判卡**：记录作者、体裁、版本、写作背景和局限；`agent_suggested` 与 `user_confirmed` 明确区分。

## 已知限制

- SQLite FTS5 的默认 tokenizer 对中文分词有限；中文检索建议使用明确人名、地名、年代或英文术语，向量检索可以补强语义召回。
- OCR 是可选依赖，未安装时图片和扫描件会给出明确安装提示；含文字层的 PDF、TXT、Markdown 不受影响。
- 证据阅读器会按保存的 PDF bbox 在页面上画出黄色半透明高亮；非 PDF 的 TXT/Markdown 则显示保存的原文段落。
