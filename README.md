# 法律条文智能检索服务 (Legal Provision Search API)

本项目是一个基于 FastAPI 和 RAG（检索增强生成）架构的法律条文智能检索系统。系统采用**三阶段检索架构**（LLM 动态路由挑选文件 -> Numpy 本地向量余弦初排 -> 多线程调用外部 Reranker 精排），支持通过 API 提供后台检索服务，也支持在终端通过命令行交互式检索。

## 1. 环境准备

在启动服务之前，请确保已安装 Python 3.8+，并安装了所有依赖包。建议使用虚拟环境：

```bash
# 创建虚拟环境 (如尚未创建)
python -m venv .venv

# 激活虚拟环境 (Linux / macOS)
source .venv/bin/activate
# Windows 用户请使用: .venv\Scripts\activate

# 安装核心依赖
pip install -r requirements.txt
```

*注意：请确保项目根目录下的 `config.py` (或 `.env`) 中已正确配置了 LLM、Embedding 以及 Reranker 相关的 API Key 和 Base URL。同时确保 `database/legal_provisions/` 目录下存放了向量化好的 `.json` 法律文件。*

## 2. 交互式命令行检索 (CLI 模式)

如果您只需要在本地测试检索效果，无需启动 Web 服务，可以直接运行项目提供的命令行入口：

```bash
python main.py
```

启动后，系统会进入交互式问答模式。您可以输入多个关键词（用逗号分隔），并自定义返回结果数量，系统会实时打印各阶段（LLM 路由、向量检索、精排）的耗时与最终法条匹配结果。

## 3. 后台启动 FastAPI 检索服务

在 Linux 或 macOS 服务器上，可以使用 `nohup` 命令将 API 服务挂载到后台持续运行，即使关闭终端终端服务也不会中断。

在项目根目录下执行以下命令：

```bash
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

**命令说明：**
- `nohup`：让程序忽略终端挂断信号（SIGHUP），在后台持续运行。
- `uvicorn server:app`：启动 `server.py` 文件中的 `app`（FastAPI 实例）。
- `--host 0.0.0.0 --port 8000`：绑定所有可用的网络接口，并指定监听端口为 8000。
- `> uvicorn.log 2>&1`：将标准输出和错误输出合并重定向到 `uvicorn.log` 文件。
- `&`：将该任务放到后台执行。

## 4. 查看运行与性能日志

本项目内置了规范化的日志系统（包含文件大小自动轮转功能）。核心业务日志、耗时统计以及 Uvicorn 访问日志都会自动写入到 `logs` 目录中：

- **服务端日志**：`logs/app.log`
- **命令行界面日志**：`logs/cli_app.log`

**实时查看 API 服务日志滚动输出：**
```bash
tail -f logs/app.log
```
*(按下 `Ctrl + C` 退出实时查看，服务仍会在后台运行)*

## 5. 测试 API 接口

服务启动后，可以通过 `curl` 或者 Postman 发送 POST 请求，测试检索服务是否正常：

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "db_type": "legal_provisions",
  "query": "关于抢劫罪的处罚标准，入室抢劫",
  "top_k": 5,
  "score": 0.5
}'
```

**预期的 JSON 响应格式**：
```json
{
  "code": 200,
  "msg": "success",
  "data": [
    {
      "法条": "中华人民共和国刑法 第二编 分则 第五章 侵犯财产罪 第二百六十三条",
      "内容": "以暴力、胁迫或者其他方法抢劫公私财物的，处三年以上十年以下有期徒刑...",
      "相似度": 0.9852
    }
  ]
}
```

## 6. 停止后台服务

如果需要重启或停止该后台服务，需要先找到该进程的 PID，然后将其终止。

**步骤 1：查找运行在 8000 端口的进程号（PID）**
```bash
lsof -i:8000
```
*(如果没有 `lsof` 命令，也可以使用 `ps aux | grep uvicorn` 查找)*

**步骤 2：终止进程**
找到对应的 PID（第二列），执行以下命令强制终止：
```bash
kill -9 <你的PID>
```
*(例如：如果查到的 PID 是 12345，则执行 `kill -9 12345`)*