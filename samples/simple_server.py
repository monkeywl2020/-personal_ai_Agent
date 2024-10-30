
import sys
import os
import asyncio
from typing import Generator
import json
from fastapi import FastAPI, Request, HTTPException

# 将项目的根目录添加到 sys.path 中
a = os.path.abspath(__file__)
#print(a)
b = os.path.dirname(a)
#print(b)
#sys.path.append(b)
c = os.path.dirname(b)
#print(c)

sys.path.append(c)
print(sys.path)

from aiAgent.msg.message import Msg
from aiAgent.agents.base_agent import BaseAgent
from aiAgent.pipelines.functional import sequentialpipeline
from aiAgent.models.model_response import ModelResponse
from aiAgent.pipelines.functional import whilelooppipeline

app = FastAPI()

config_list = [
    {
        "model": "/work/wl/.cache/modelscope/hub/qwen/Qwen2-72B-Instruct-GPTQ-Int4",
        "base_url": "http://172.21.30.230:8980/v1",
        "api_type": "qwen",
        "api_key": "NotRequired",
        "max_tokens":2048
    }
]
#        "api_type": "openai",
llm_config = {
    "config_list": config_list,
}

chat_agent = BaseAgent(
    name="wlchat1",
    llm_config=llm_config,
    max_num_auto_reply = 2,
    sys_prompt = "你是一个友好的机器人助手，协助用户解决问题",
    #human_input_mode = "ALWAYS"
)


@app.post("/v1/chat/completions")
async def receive_message(request: Request):
    # 处理接收到的消息
    try:
        # 尝试使用 request.json() 解析请求体
        request_content = await request.json()
        response = process_message(request_content)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format: Not an object")
    
    return response

def process_message(request_content):
    # 在这里添加处理消息的逻辑
    # 例如，调用你现有的代码来处理消息
    message_content = request_content.get("content", "") 
    x = Msg(name="user",role="user", content=message_content)
    rsp = chat_agent(x)
    print("=================process_message=========================")
    print(rsp)
    print("=================process_message=========================")
    return rsp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7856)

