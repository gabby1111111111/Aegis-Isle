#!/usr/bin/env python3
"""
简化版 AegisIsle 演示程序
这个版本只包含核心功能，便于快速体验和理解
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from pathlib import Path
import json
import time

# 创建FastAPI应用
app = FastAPI(
    title="AegisIsle 演示版",
    description="多智能体协同RAG系统演示",
    version="0.1.0"
)

# 数据模型
class QueryRequest(BaseModel):
    query: str
    max_docs: int = 5
    use_agents: bool = False

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

# 简单的内存存储
documents_store = []  # 存储上传的文档
knowledge_base = {}   # 模拟知识库

# 模拟的智能体类
class SimpleAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    def process(self, query: str) -> str:
        if self.role == "researcher":
            return f"🔍 研究员 {self.name}: 正在深度研究「{query}」相关信息..."
        elif self.role == "retriever":
            return f"📚 检索员 {self.name}: 正在查找「{query}」相关文档..."
        elif self.role == "summarizer":
            return f"📝 总结员 {self.name}: 正在总结「{query}」的分析结果..."
        else:
            return f"🤖 智能体 {self.name}: 正在处理「{query}」..."

# 创建演示智能体
agents = {
    "researcher": SimpleAgent("小研", "researcher"),
    "retriever": SimpleAgent("小档", "retriever"),
    "summarizer": SimpleAgent("小结", "summarizer"),
}

@app.get("/", response_class=HTMLResponse)
async def home():
    """主页 - 显示演示界面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AegisIsle 演示系统</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                padding: 30px;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            h1 { text-align: center; margin-bottom: 30px; }
            .demo-box {
                background: rgba(255,255,255,0.2);
                padding: 20px;
                margin: 20px 0;
                border-radius: 10px;
            }
            button {
                background: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
            }
            button:hover { background: #45a049; }
            input, textarea {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .result {
                background: rgba(0,0,0,0.3);
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 AegisIsle 多智能体协同RAG系统演示</h1>

            <div class="demo-box">
                <h2>📚 1. 文档上传测试</h2>
                <p>上传文档到知识库（目前支持文本文件）</p>
                <input type="file" id="fileInput" accept=".txt,.md">
                <button onclick="uploadFile()">上传文档</button>
                <div id="uploadResult" class="result"></div>
            </div>

            <div class="demo-box">
                <h2>🤖 2. 智能问答测试</h2>
                <p>输入问题，体验RAG智能问答</p>
                <textarea id="queryInput" rows="3" placeholder="请输入你的问题，例如：什么是人工智能？"></textarea>
                <br>
                <label><input type="checkbox" id="useAgents"> 使用多智能体协同</label>
                <br><br>
                <button onclick="askQuestion()">提交问题</button>
                <div id="queryResult" class="result"></div>
            </div>

            <div class="demo-box">
                <h2>📊 3. 系统状态</h2>
                <button onclick="getSystemInfo()">查看系统信息</button>
                <div id="systemInfo" class="result"></div>
            </div>
        </div>

        <script>
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                if (!file) {
                    alert('请选择文件');
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/api/v1/documents/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    document.getElementById('uploadResult').textContent =
                        `上传结果：\\n${JSON.stringify(result, null, 2)}`;
                } catch (error) {
                    document.getElementById('uploadResult').textContent =
                        `错误：${error.message}`;
                }
            }

            async function askQuestion() {
                const query = document.getElementById('queryInput').value;
                const useAgents = document.getElementById('useAgents').checked;

                if (!query.trim()) {
                    alert('请输入问题');
                    return;
                }

                try {
                    const response = await fetch('/api/v1/query/', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: query,
                            use_agents: useAgents
                        })
                    });
                    const result = await response.json();
                    document.getElementById('queryResult').textContent =
                        `回答：\\n${result.answer}\\n\\n详细信息：\\n${JSON.stringify(result, null, 2)}`;
                } catch (error) {
                    document.getElementById('queryResult').textContent =
                        `错误：${error.message}`;
                }
            }

            async function getSystemInfo() {
                try {
                    const response = await fetch('/api/v1/system/info');
                    const result = await response.json();
                    document.getElementById('systemInfo').textContent =
                        JSON.stringify(result, null, 2);
                } catch (error) {
                    document.getElementById('systemInfo').textContent =
                        `错误：${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/api/v1/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """文档上传接口"""
    try:
        # 读取文件内容
        content = await file.read()

        # 模拟文档处理
        doc_info = {
            "id": f"doc_{len(documents_store) + 1}",
            "filename": file.filename,
            "size": len(content),
            "content": content.decode('utf-8') if file.filename.endswith(('.txt', '.md')) else "二进制文件",
            "upload_time": time.time()
        }

        # 存储文档
        documents_store.append(doc_info)

        # 模拟添加到知识库
        knowledge_base[doc_info["id"]] = {
            "content": doc_info["content"],
            "metadata": {
                "filename": file.filename,
                "size": doc_info["size"]
            }
        }

        return {
            "success": True,
            "message": f"文档 {file.filename} 上传成功！",
            "document_id": doc_info["id"],
            "processed_content_length": len(doc_info["content"]),
            "total_documents": len(documents_store)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档上传失败：{str(e)}")

@app.post("/api/v1/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """智能问答接口"""
    start_time = time.time()

    try:
        # 模拟检索过程
        retrieved_docs = []
        for doc_id, doc_data in knowledge_base.items():
            # 简单的关键词匹配
            if any(word.lower() in doc_data["content"].lower() for word in request.query.split()):
                retrieved_docs.append({
                    "document_id": doc_id,
                    "content": doc_data["content"][:200] + "...",
                    "score": 0.8,
                    "metadata": doc_data["metadata"]
                })

        # 模拟智能体处理
        agent_responses = []
        if request.use_agents:
            for agent_name, agent in agents.items():
                agent_responses.append(agent.process(request.query))

        # 生成回答
        if retrieved_docs:
            answer = f"""基于知识库的回答：

根据检索到的{len(retrieved_docs)}个相关文档，我来回答你的问题「{request.query}」：

{retrieved_docs[0]["content"]}

这是基于文档内容的分析结果。"""
        else:
            answer = f"""抱歉，在当前知识库中没有找到关于「{request.query}」的相关信息。

建议：
1. 上传相关文档到知识库
2. 尝试使用不同的关键词重新提问
3. 检查问题的表述是否清晰"""

        if agent_responses:
            answer += f"\n\n🤖 多智能体协同处理结果：\n" + "\n".join(agent_responses)

        processing_time = time.time() - start_time

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=retrieved_docs,
            processing_time=processing_time,
            metadata={
                "total_documents_searched": len(knowledge_base),
                "documents_found": len(retrieved_docs),
                "agents_used": len(agent_responses) if request.use_agents else 0,
                "processing_steps": [
                    "文档检索",
                    "智能体协同" if request.use_agents else "直接生成",
                    "答案合成"
                ]
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询处理失败：{str(e)}")

@app.get("/api/v1/system/info")
async def get_system_info():
    """系统信息接口"""
    return {
        "system_name": "AegisIsle 演示版",
        "version": "0.1.0",
        "status": "running",
        "capabilities": {
            "document_upload": True,
            "intelligent_qa": True,
            "multi_agent_collaboration": True,
            "rag_processing": True
        },
        "statistics": {
            "total_documents": len(documents_store),
            "knowledge_base_size": len(knowledge_base),
            "active_agents": len(agents),
            "supported_formats": ["txt", "md"]
        },
        "agents": {
            name: {"role": agent.role, "name": agent.name}
            for name, agent in agents.items()
        }
    }

@app.get("/api/v1/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "AegisIsle Demo",
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    print("启动 AegisIsle 演示系统...")
    print("演示功能：")
    print("   - 文档上传和存储")
    print("   - 智能问答系统")
    print("   - 多智能体协同")
    print("   - RAG检索增强")
    print("访问地址: http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000)