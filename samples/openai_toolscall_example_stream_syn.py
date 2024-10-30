
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
            "model": "/work/wl/.cache/modelscope/hub/qwen/Qwen2-72B-Instruct-GPTQ-Int4",
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
            "api_key": "sk-proj-xxx",
            "stream": True,
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
        """根据给的地址获取天气信息,地址中文名"""
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

    x = Msg(name="user",role="user", content=content)
    print("-----------x:",x)
    # 指定 api_type 为 openai
    x = ag1.stream(x, use_tools=True, api_type = "openai")
    # 如果设置为流式，就获取出来的是生成器
    if isinstance(x,Generator):
        print("----------1-Generator:")
        for chunk in x:
            pass
        #    print("------------------- Start of Chunk -------------------",flush=True)
        #    print(chunk,flush=True)
        #    print("------------------- End of Chunk -------------------",flush=True)

        #这个是所有的消息内容
                
        x = Msg(name=ag1.name ,role="assistant", content=chunk)
        print("------------------- End of Chunk ------------------allmsg:\n",x,flush=True)    

    else:
        print("--ag1-----------------")
        print(x)
        print("--ag1-----------------")


    #'''

    # 上面生成了 function call的响应，下面需要将消息转发给 llm
    # 将函数转换为工具,注册函数 
    ag2.register_tool_for_llm(get_current_weather)
    # 没配置 llm_config，所以不会调用 llm，不会有流式输出
    x = ag2.generate_response(x)
    print("--ag2---------x:",x)


    #
    x = ag1.stream(x, use_tools=True, api_type = "openai")
    # 如果设置为流式，就获取出来的是生成器
    if isinstance(x,Generator):
        print("----------1-Generator:")
        for chunk in x:
            pass
        #    print("------------------- Start of Chunk -------------------",flush=True)
        #    print(chunk,flush=True)
        #    print("------------------- End of Chunk -------------------",flush=True)

        #这个是所有的消息内容
                
        x = Msg(name=ag1.name ,role="assistant", content=chunk)
        print("------------------- End of Chunk ------------------allmsg:\n",x,flush=True)    

    else:
        print("--ag1-----------------")
        print(x)
        print("--ag1-----------------")

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