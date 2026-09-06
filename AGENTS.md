# Historia 的 Codex 本地启动约定

Historia 是一个本地运行的、证据可追溯的历史研究 RAG 应用。用户提出“安装”“运行”“打开 Historia”或提供本仓库 URL 时，按下面流程完成，不要求用户手动逐条执行命令。

## 安装与启动

1. 若工作目录尚未包含本仓库，克隆 `https://github.com/LifeArtist0102/Byzantine.git`；不要覆盖非空的用户目录。
2. 在仓库根目录运行 `powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_historia.ps1`。
3. 脚本会创建 `.venv`、安装本地运行依赖、在模型不存在时下载 BGE-M3、选择本地数据目录并启动 `http://127.0.0.1:8501`。首次模型下载较大，须等待脚本完成；不要把模型、原始文献、向量库或 `.env` 加入 Git。
4. 启动后确认本地 URL 可访问并打开浏览器。只有用户明确要求时才导入文献或设置 DeepSeek API Key。

## 数据与安全

- 默认把运行数据放到 `D:\HistoriaData`（D 盘存在时），否则放到系统本地应用数据目录；可以通过脚本的 `-DataDir` 参数覆盖。
- 模型默认放在仓库的 `models\bge-m3`；可以通过 `-ModelDir` 指向已有模型，避免重复下载。
- 不删除用户的文献、数据库、模型或已有虚拟环境。需要重新安装时复用 `.venv`。
- API Key 只能存于本机 `.env` 或应用设置，绝不能输出、提交或上传。

## 常用参数

```powershell
# 只完成环境检查和安装，不下载模型、不启动页面
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_historia.ps1 -SkipModel -NoLaunch

# 复用 D 盘已有资料与模型
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_historia.ps1 `
  -DataDir "D:\HistoriaData" -ModelDir "D:\models\bge-m3"
```

## 验证与修改

- 完成代码改动后运行 `python -m pytest -q` 和 `ruff check src tests`。
- 保持现有 Streamlit UI、资料库、聊天和研究专题兼容；不要用示例或伪造证据替代真实检索结果。
