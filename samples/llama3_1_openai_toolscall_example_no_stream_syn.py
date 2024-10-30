
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
            "api_key": "sk-proj-xxxx",
        },
        {
            "model": "/work/wl/wlwork/my_models/meta-llama-3___1-70b-instruct-gptq-int4",
            "base_url": "http://172.21.30.230:8981/v1",
            "api_type": "llama",
            "api_key": "EMPTY",
        }
    ]

    llm_config = {
        "config_list": config_list,
    }

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

    #--------------------------------------------------
    # 1：定义function call用到的函数
    #--------------------------------------------------
    # 查询心理测试 所有类别
    def query_scale_category()-> dict:
        """ 查询心理测试包含的所有类别, 返回所有的心理测试的类别名称和对应的类别ID """
        status = "success"
        message = ""
        data = [{'categoryName': '临床', 'categoryId': '1777177860138758145'},
        {'categoryName': '小学', 'categoryId': '1777179115410059266'},
        {'categoryName': '小学', 'categoryId': '1777220002819239937'}, 
        {'categoryName': '非临床', 'categoryId': '1777177939075559426'}, 
        {'categoryName': '焦虑', 'categoryId': '1777178614488526849'}, 
        {'categoryName': '中学', 'categoryId': '1777179158028382209'}, 
        {'categoryName': '成人', 'categoryId': '1777180748089028610'}, 
        {'categoryName': '成人', 'categoryId': '1777184964689948674'}, 
        {'categoryName': '偏执', 'categoryId': '1777184596287451138'}, 
        {'categoryName': '社交恐惧', 'categoryId': '1777180622289268737'},
        {'categoryName': '其他', 'categoryId': '1777219935232225281'}]    

        return {"status": status, "message": message, "data": data}

    # 根据心理测试类别id 查询心理测试表格
    def query_scale_list(categoryId: str) -> dict:
        """ 根据心理测试的类别id,查询此类别有哪些心理测试的表格,返回测试表格id:scaleIdid """
        if categoryId is None or categoryId == "":
            status = "failed"
            message = "category_id is null"
            data = None
        else:
            status = "success"
            message = ""
            data =  [{'scaleName': '青少年焦虑多维量表MASC', 'scaleId': '469078509608763393'}]
    
        return {"status": status, "message": message, "data": data}

    # 根据用户的 userid 和 表格id,创建一个心里测试任务
    def create_pa_test_task(user_id: str, scaleId: str) -> dict:
        """ 根据用户的 user_id 和 表格scaleIdid,创建一个心里测试任务,进行心理量表的测评. """
        if user_id is None or user_id == "":
            status = "failed"
            message = "user_id is null"
            data = None
        elif scaleId is None or scaleId == "":
            status = "failed"
            message = "scaleId is null"
            data = None
        else:
            status = "success"
            message = ""
            data = {'taskId': '476705551993012224'}
            
        return {"status": status, "message": message, "data": data, "type":"CREATE_PA"}


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



    # 将函数转换为工具,注册函数
    #ag1.register_tool_for_llm(get_current_weather)
    ag1.register_tool_for_llm(query_scale_category)

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
    #content = "当前武汉天气怎么样,北京的呢?深圳?"

    #------------------------------------
    #  ******  单function call  ****
    #
    #------------------------------------
    #content = "当前深圳天气怎么样?"
    #content = "Question:你支持的function call,可以使用openAIclient么？为什么多次调用后返回的结果不一样？"
    #content = "你好？我是wl"
    content = "心理测试有哪些类别？"
    x = Msg(name="user",role="user", content=content)
    print("-----------x:",x)
    # 指定 api_type 为 openai
    x = ag1.generate_response(x, use_tools=True, api_type = "llama")
    print("--ag1-----------------")
    print(x)
    print("--ag1-----------------")

    #'''

    '''
    # 上面生成了 function call的响应，下面需要将消息转发给 llm
    # 将函数转换为工具,注册函数
    ag2.register_tool_for_llm(get_current_weather)
    x = ag2.generate_response(x)
    print("--ag2---------x:",x)


    #
    x = ag1.generate_response(x, use_tools=True, api_type = "openai")
    print("--ag1-----------------")
    print(x)
    print("--ag1-----------------")
    '''
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