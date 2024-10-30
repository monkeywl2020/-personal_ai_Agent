
import sys
import os
import json
import asyncio
from typing import Generator

# 将项目的根目录添加到 sys.path 中
a = os.path.abspath(__file__)
print(a)
b = os.path.dirname(a)
print(b)
c = os.path.dirname(b)
print(c)

sys.path.append(c)
print(sys.path)

from aiAgent.msg.message import Msg
from aiAgent.agents.base_agent import BaseAgent
from aiAgent.pipelines.functional import sequentialpipeline
from aiAgent.models.model_response import ModelResponse
from aiAgent.pipelines.functional import whilelooppipeline


def main():
    config_list = [
        {
            "model": "/work/wl/wlwork/my_models/Qwen2___5-72B-Instruct-GPTQ-Int4",
            "base_url": "http://172.21.30.230:8980/v1",
            "api_type": "qwen",
            "api_key": "NotRequired",
            "max_tokens":2048,
            #"hide_toolscall":True,
        }
        ,
        {
            "model": "gpt-4o",
            "api_type": "openai",
            "api_key": "sk-proj-xxxx",
        }
    ]

    llm_config = {
        "config_list": config_list,
    }


    # "stream": False 流式输出不由模型配置控制，由调用 函数控制

    ag1 = BaseAgent(
        name="wltest1",
        llm_config=llm_config,
        max_num_auto_reply = 2,
        #sys_prompt = "你是一个老师负责给学生答疑解惑",
        #human_input_mode = "ALWAYS"
    )

    # 用户agent： user proxy,不使用llm
    ag2 = BaseAgent(
        name="wltest2",
        #llm_config=llm_config,
        #sys_prompt = "你是一个小学生,对老师提问",
        human_input_mode = "ALWAYS"
    )

    # 一个简单的函数，用于获取天气信息
    def get_current_weather(location, unit='celsius'):
        """根据给的地址获取天气信息,地址中文名,参数 unit 默认是celsius  """
        if '武汉' in location:
            return json.dumps({'location': '武汉', 'temperature': '10', 'unit': 'celsius'})
        elif '北京' in location:
            return json.dumps({'location': '北京', 'temperature': '30', 'unit': 'celsius'})
        elif '深圳' in location:
            return json.dumps({'location': '深圳', 'temperature': '22', 'unit': 'celsius'})
        else:
            return json.dumps({'location': location, 'temperature': 'unknown'})

    # 将函数转换为工具,注册函数
    ag1.register_tool_for_llm(get_current_weather)

    #    基础接口例子
    #------------------------------------
    #  0：非流式（同步）默认是异步
    #
    #------------------------------------
    #'''
    #------------------------------------
    #  ******  并发多function call  ****
    #
    #------------------------------------
    content = "当前武汉天气怎么样,北京的呢?深圳?"

    #------------------------------------
    #  ******  单function call  ****
    #
    #------------------------------------
    #content = "当前深圳天气怎么样?"
    #content = "我是wl,你是千问吗?"

    #--------------------------------------------------
    #  用户注册到 agent中去的用来显示流式处理的回调函数。 
    #  当然可以在这里面往外发消息也可以。
    #   使用示例  stream_to_client 打印收到的流式消息
    #--------------------------------------------------
    def stream_msg_to_client(text):
        print("stream_to_client--- Start of Chunk -------------------",flush=True)
        print(text,flush=True)
        print("stream_to_client--- End of Chunk -------------------",flush=True)

    x = Msg(name="user",role="user", content=content)
    print("-----------x:",x)
    # 指定 api_type 为 openai
    x = asyncio.run(ag1.a_stream(x, use_tools=True, api_type = "qwen", stream_callback = stream_msg_to_client))
    print("--ag1-----------------")
    print(x)
    print("--ag1-----------------")

    #'''

    # 上面生成了 function call的响应，下面需要将消息转发给 llm
    # 将函数转换为工具,注册函数
    #'''
    ag2.register_tool_for_llm(get_current_weather)
    x = ag2(x)
    print("--ag2---------x:",x)


    
    x = asyncio.run(ag1.a_stream(x, use_tools=True, api_type = "qwen", stream_callback = stream_msg_to_client))
    print("--ag1-----------------")
    print(x)
    print("--ag1-----------------")


#主函数
if __name__ == "__main__":
    main()