
import sys
import os
import asyncio
from typing import Generator

# 将项目的根目录添加到 sys.path 中
a = os.path.abspath(__file__)
print(a)
b = os.path.dirname(a)
print(b)
#sys.path.append(b)
c = os.path.dirname(b)
print(c)
#d = os.path.dirname(c)
#print(d)
#sys.path.append(d) # 添加 botAssistant/botAssistant
#e = os.path.dirname(d)
#print(e)  # 添加最外层 botAssistant
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
            "model": "/work/wl/.cache/modelscope/hub/qwen/Qwen2-72B-Instruct-GPTQ-Int4",
            "base_url": "http://172.21.30.230:8980/v1",
            "api_type": "qwen",
            "api_key": "NotRequired",
            "max_tokens":2048,
            #"hide_toolscall":True,
        },
        {
            "model": "gpt-4o",
            "api_type": "openai",
            "api_key": "sk-proj-xxxxx",
        }
    ]
    #        "api_type": "openai",
    llm_config = {
        "config_list": config_list,
    }


    # "stream": False 流式输出不由模型配置控制，由调用 函数控制

    ag1 = BaseAgent(
        name="wltest1",
        llm_config=llm_config,
        max_num_auto_reply = 2        
        #sys_prompt = "你是一个老师负责给学生答疑解惑",
        #human_input_mode = "ALWAYS"
    )

    ag2 = BaseAgent(
        name="wltest2",
        llm_config=llm_config,
        sys_prompt = "你是一个小学生,对老师提问",
        human_input_mode = "NEVER"
    )

    #    基础接口例子
    #------------------------------------
    #  0：非流式（同步）默认是异步
    #
    #------------------------------------
    #'''
    content = {"content": "我是wl,你是谁？"}
    x = Msg(name="user",role="user", content=content)
    print("-----------x:",x)
    x = ag1.generate_response(x,api_type = "openai")
    print("-------------------")
    print(x)
    #print("-------------------")
    #'''

    #------------------------------------
    #  1：非流式（异步）默认异步
    #
    #------------------------------------
    '''
    content = {"content": "我是wl,你是谁？"}
    x = Msg(name="user",role="user", content=content)
    print("-----------x:",x)
    x = ag1(x)
    print("-------------------")
    print(x)
    #print("-------------------")
    '''

    #------------------------------------
    #   2：流式，同步
    #
    #------------------------------------
    '''
    content = {"content": "我是谁？"}
    x = Msg(name="user",role="user", content=content)
    print("----------1-x:",x)
    x = ag1.stream(x)

    # 如果设置为流式，就获取出来的是生成器
    if isinstance(x,Generator):
        print("----------1-Generator:")
        for chunk in x:
            pass
        #    print("------------------- Start of Chunk -------------------",flush=True)
        #    print(chunk,flush=True)
        #    print("------------------- End of Chunk -------------------",flush=True)

        #这个是所有的消息内容
                
        allmsg = Msg(name=ag1.name ,role="assistant", content=chunk)
        print("------------------- End of Chunk ------------------allmsg:\n",allmsg,flush=True)    

    else:
        print("-----------x:",x)
    '''

    #------------------------------------
    #    3：  流式，异步
    #
    #------------------------------------
    '''
    content = {"content": "你支持function call 或者 tool call么？"}
    x = Msg(name="user",role="user", content=content)
    print("----------1-x:",x)
    x = asyncio.run(ag1.a_stream(x))

    # 如果设置为流式，就获取出来的是生成器
    if isinstance(x,Generator):
        print("----------1-Generator:")
        for chunk in x:
            pass
            #print("------------------- Start of Chunk -------------------",flush=True)
            #print(chunk,flush=True)
            #print("------------------- End of Chunk -------------------",flush=True)

        #这个是所有的消息内容
        allmsg = Msg(name=ag1.name ,role="assistant", content=chunk)
        print("------------------- End of Chunk ------------------allmsg:\n",allmsg,flush=True)    

    else:
        print("-----------x:",x)
    '''

    '''
    x = ag2(allmsg)
    # 如果设置为流式，就获取出来的是生成器
    if isinstance(x,Generator):
        print("----------2-Generator:")
        for chunk in x:
            print("------------------- Start of Chunk -------------------",flush=True)
            print(chunk,flush=True)
            print("------------------- End of Chunk -------------------",flush=True)

        allmsg = Msg(name=ag1.name ,role="assistant", content=chunk)
        print("------------------- End of Chunk -----------------2-allmsg:\n",allmsg,flush=True)    

    else:
        print("----------2-x:",x)
    '''

    '''
    if isinstance(x,ModelResponse):
        print("-----------Generator:")
        for chunk in x._stream:
            print("------------------- Start of Chunk -------------------",flush=True)
            print(chunk,flush=True)
            print("------------------- End of Chunk -------------------",flush=True)

    else:
        print("-----------x:",x)
    '''

    '''
    ag2 = BaseAgent(
        name="wltest2",
        llm_config=llm_config,
    )

    content = {"content": "你是谁？"}
    x = Msg(name="user",role="user", content=content)

    x = sequentialpipeline([ag1,ag2],x)

    #agent_with_number开始，通过message与 agent_guess_number 对话
    '''
    #------------------------------------
    #    3：  while 循环，人工干预
    #
    #------------------------------------
    '''
    content = {"content": "请开始提问"}
    x = Msg(name="user",role="user", content=content)
    print("-----------x:",x)
    # whilelooppipeline 循环使用,如果消息为空或者 循环次数超过5次则返回
    whilelooppipeline([ag2,ag1], (lambda i,x: i<5 and bool(x)), x)
    '''

#主函数
if __name__ == "__main__":
    main()