# 法律智能检索服务 (Legal Search API)

本项目是一个基于 FastAPI + RAG 的法律检索系统，现已支持两类数据检索：

- **法律条文检索**（`legal_provisions`）
- **法律案例检索**（`legal_cases`）

系统采用三阶段检索架构：

1. **LLM 动态路由**：从本地库中筛选可能相关的数据文件  
2. **Numpy 向量初排**：基于余弦相似度快速召回  
3. **Reranker 精排**：多线程调用外部重排模型做相关性优化

同时支持 API 服务模式和本地 CLI 交互模式。

---

## 1. 环境准备

请确保 Python 3.8+，建议使用虚拟环境：

```bash
# 创建虚拟环境（如尚未创建）
python -m venv .venv

# 激活虚拟环境（Linux / macOS）
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

请确保 `config.py`（或 `.env`）中已正确配置以下模型参数：

- LLM（路由/抽取）
- Embedding（向量化/召回）
- Reranker（精排）

并确保本地数据目录存在并包含向量化后的 JSON 文件，例如：

- `database/legal_provisions/`（法律条文库）
- `database/legal_case/`（法律案例库，含 `违法行为向量` 字段）

---

## 2. CLI 模式（本地交互检索）

### 2.1 法律条文检索 CLI
```bash
python main.py
```

### 2.2 法律案例检索 CLI
```bash
python main_case_search.py
```

启动后可输入多个关键词（中英文逗号分隔），并设置返回数量等参数。

---

## 3. 启动 FastAPI 服务

在项目根目录运行：

```bash
nohup uvicorn server:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &
```

参数说明：

- `nohup`：后台持续运行
- `uvicorn server:app`：启动 `server.py` 的 `app`
- `--host 0.0.0.0 --port 8000`：监听全部网卡，端口 8000
- `> uvicorn.log 2>&1`：合并输出到日志文件
- `&`：后台执行

---

## 4. 日志说明

项目使用统一日志规范与轮转策略，默认输出：

- 服务端日志：`logs/app.log`
- 条文 CLI 日志：`logs/cli_app.log`
- 案例 CLI 日志（若按示例实现）：`logs/cli_case_search.log`

实时查看服务日志：

```bash
tail -f logs/app.log
```

---

## 5. API 使用说明

统一接口：

- `POST /search`

请求体：

```json
{
  "db_type": "legal_provisions",
  "query": "关于抢劫罪的处罚标准，入室抢劫",
  "top_k": 5,
  "score": 0.5
}
```

`db_type` 支持：

- `legal_provisions`：法律条文检索
- `legal_cases`：法律案例检索

---

### 5.1 法律条文检索示例

```bash
curl -X POST 'http://127.0.0.1:8000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "db_type": "legal_provisions",
  "query": "关于抢劫罪的处罚标准，入室抢劫",
  "top_k": 5,
  "score": 0.5
}'
```

示例响应：

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

---

### 5.2 法律案例检索示例

```bash
curl -X POST 'http://127.0.0.1:8000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "db_type": "legal_cases",
  "query": "虚构事实骗取财物，隐瞒真相，电信网络诈骗",
  "top_k": 5,
  "score": 0.5
}'
```

示例响应：

```json
{
  "code": 200,
  "msg": "success",
  "data": [
    {
      "标题": "宋某岩诈骗案（电信网络诈骗犯罪中的诈骗数额认定）",
      "案号": "（2021）苏0508刑初912号",
      "关键词": ["刑事", "诈骗罪", "诈骗数额", "犯罪成本"],
      "裁判原则": "......",
      "基本案情": "......",
      "裁判理由": "......",
      "法律适用": "......",
      "违法行为": "虚构事实骗取财物，隐瞒真相骗取财物，组织实施电信网络诈骗",
      "相似度": 0.9321,
      "命中关键词": "电信网络诈骗"
    }
  ]
}
```

---

## 6. 停止后台服务

### 6.1 查询 PID
```bash
lsof -i:8000
```

或：

```bash
ps aux | grep uvicorn
```

### 6.2 结束进程
```bash
kill -9 <PID>
```

---

## 7. 常见问题

### Q1：Kimi 模型报错 `invalid temperature: only 1 is allowed for this model`
请将对应 `ChatOpenAI` 的 `temperature` 设置为 `1`。

### Q2：日志里出现 `Retrying request to /chat/completions ...`
通常是临时网络抖动/限流导致的 SDK 自动重试，若后续出现 `200 OK` 即为正常。
可通过降低并发减少重试频率。

### Q3：案例检索无结果
请检查案例库 JSON 中是否包含：

- `违法行为`（字符串）
- `违法行为向量`（长度等于 `EMBEDDING_DIMENSIONS` 的向量）