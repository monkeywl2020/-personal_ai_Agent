
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
'''
        {
            "model": "gpt-4o",
            "api_type": "openai",
            "api_key": "sk-proj-xxxx",
        }
'''

def main():
    config_list = [
        {
            "model": "/work/wl/wlwork/my_models/Qwen2___5-72B-Instruct-GPTQ-Int4",
            "base_url": "http://172.21.30.230:8980/v1",
            "api_type": "qwen",
            "api_key": "NotRequired",
            "max_tokens":2048,
        }
        ,
        {
            "model": "/work/wl/wlwork/my_models/Qwen2___5-72B-Instruct-GPTQ-Int4",
            "base_url": "http://172.21.30.230:8980/v1",
            "api_type": "qwen-openai",
            "api_key": "EMPTY",
            "max_tokens":2048,
        }
    ]

    llm_config = {
        "config_list": config_list,
    }


    # "stream": False 流式输出不由模型配置控制，由调用 函数控制
    # hide_toolscall 参数的意思是隐藏 toolcalls，当有tool call的时候agent会自动调用，不需要用户干预
    # 默认是 False，不隐藏。也就是llm生成了tool calls消息一样的当普通内容发送给 下一个处理者。
    # 如果设置为True，llm生成的tool calls消息不会发送给下一个处理者，而是自动调用函数，并且处理成最终响应发出给下一个处理者。
    ag1 = BaseAgent(
        name="wltest1",
        llm_config=llm_config,
        max_num_auto_reply = 2,
        hide_toolscall=True,
        #sys_prompt = "你是一个老师负责给学生答疑解惑",
        #human_input_mode = "ALWAYS"        
    )

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
        {'categoryName': '非临床', 'categoryId': '1777177939075559426'}, 
        {'categoryName': '焦虑', 'categoryId': '1777178614488526849'}, 
        {'categoryName': '中学', 'categoryId': '1777179158028382209'}, 
        {'categoryName': '成人', 'categoryId': '1777180748089028610'}, 
        {'categoryName': '偏执', 'categoryId': '1777184596287451138'}, 
        {'categoryName': '社交恐惧', 'categoryId': '1777180622289268737'},
        {'categoryName': '其他', 'categoryId': '1777219935232225281'}]    

        return {"status": status, "message": message, "data": data}

    # 根据心理测试类别id 查询心理测试表格
    def query_scale_list(categoryId: str) -> dict:
        """ 根据心理测试的类别id,查询此类别拥有的心理测试的表格,并返回测试表格id也就是表格的scaleIdid """
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
        """ 根据用户的 user_id 和 表格scaleIdid,创建一个心里测试任务,使得用户可以进行心理量表的测评. """
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



    # 将函数转换为工具,注册函数
    ag1.register_tool_for_llm(query_scale_category)
    ag1.register_tool_for_llm(query_scale_list)
    ag1.register_tool_for_llm(create_pa_test_task)
    content = "我想做一个青少年焦虑的测评(用户id:123456)"
    #content = "我想做一个老人的心理测评(用户id:123456)"
    #content = "我叫wl,你叫什么名字？"

    #------------------------------------
    #  ******  并发多function call  ****
    #
    #------------------------------------
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
    #ag1.register_tool_for_llm(get_current_weather)
    #content = "当前武汉天气怎么样,北京的呢?深圳?"
    #content = "当前武汉天气怎么样?"


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
    def stream_to_client(text):
        print("stream_to_client--- Start of Chunk -------------------",flush=True)
        print(text,flush=True)
        print("stream_to_client--- End of Chunk -------------------",flush=True)

    x = Msg(name="user",role="user", content=content)
    print("-----------x:",x)
    # 指定 api_type 为 openai
    #x = ag1.generate_response(x, use_tools=True, api_type = "qwen-openai", stream_callback = stream_to_client)
    #x = ag1(x, use_tools=True, api_type = "qwen-openai", stream_callback = stream_to_client)
    #x = ag1.stream(x, use_tools=True, api_type = "qwen-openai", stream_callback = stream_to_client)
    x = asyncio.run(ag1.a_stream(x, use_tools=True, api_type = "qwen-openai", stream_callback = stream_to_client))
    # 如果设置为流式，就获取出来的是生成器
    print("--ag1-----------------")
    print(x)
    print("--ag1-----------------")
    

#主函数
if __name__ == "__main__":
    main()