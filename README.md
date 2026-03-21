# 法律条文智能检索服务 (Legal Provision Search API)

本项目是一个基于 FastAPI 和 RAG（检索增强生成）架构的法律条文智能检索服务，提供高效的本地向量检索与 LLM 重排（Rerank）功能。

## 1. 环境准备

在启动服务之前，请确保已安装 Python 3.8+，并安装了依赖包。建议使用虚拟环境：

```bash
# 创建虚拟环境 (如尚未创建)
python -m venv .venv

# 激活虚拟环境 (Linux / macOS)
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install uvicorn  # 如果 requirements.txt 中没有包含 uvicorn
```

*注意：请确保项目根目录下的 `.env` 或 `config.py` 中已正确配置了大模型与向量化相关的 API Key。*

## 2. 后台启动服务

在 Linux 或 macOS 服务器上，可以使用 `nohup` 命令将程序挂载到后台持续运行，即使关闭终端终端服务也不会中断。

在项目根目录下执行以下命令：

```bash
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

**命令说明：**
- `nohup`：让程序忽略终端挂断信号（SIGHUP），在后台持续运行。
- `server:app`：启动 `server.py` 文件中的 `app`（FastAPI 实例）。
- `--host 0.0.0.0 --port 8000`：绑定所有可用的网络接口，并指定监听端口为 8000。
- `> uvicorn.log`：将正常的控制台打印输出重定向到 `uvicorn.log` 文件。
- `2>&1`：将错误输出（stderr）也合并重定向到同一个日志文件中。
- `&`：将该任务放到后台执行。

## 3. 查看运行日志

服务启动后，所有的启动信息、请求日志以及错误堆栈都会记录在 `uvicorn.log` 中。

**实时查看日志滚动输出：**
```bash
tail -f uvicorn.log
```
*(按下 `Ctrl + C` 退出实时查看，服务仍会在后台运行)*

## 4. 测试 API 接口

服务启动后，可以通过 `curl` 或者 Postman 发送 POST 请求测试服务是否正常：

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "db_type": "legal_provisions",
  "query": "关于抢劫罪的处罚标准",
  "top_k": 5,
  "score": 0.5
}'
```

## 5. 停止后台服务

如果需要重启或停止该后台服务，需要先找到该进程的 PID，然后将其终止。

**步骤 1：查找运行在 8000 端口的进程号（PID）**
```bash
lsof -i:8000
```
*(如果没有 `lsof` 命令，也可以使用 `ps aux | grep server:app` 查找)*

**步骤 2：终止进程**
找到对应的 PID（第二列），执行以下命令强制终止：
```bash
kill -9 <你的PID>
```
*(例如：如果查到的 PID 是 12345，则执行 `kill -9 12345`)*