# -*- coding: utf-8 -*-

from typing import Any, Optional
from termcolor import colored
from ..msg.message import Msg

# 人类输入输出类，负责处理人类输入输出
class HumanIOHandler():
    #---------------------------------------------------------
    # 1： io_type 人类输入输出类型，默认是命令行
    #      
    #---------------------------------------------------------
    io_type: Optional[str] = "command_line"

    #---------------------------------------------------------
    # 初始化函数
    #      
    #---------------------------------------------------------
    def __init__(self, io_type: str, **kwargs: Any) -> None:
        self.io_type = io_type

    def input(self, prompt: str) -> str:
        if self.io_type == "command_line":
            return self.get_input_from_command_line(prompt)
        elif self.io_type == "message":
            return self.get_input_from_message(prompt)
        else:
            raise ValueError("未知的输入方式")

    def output(self, content):
        if self.io_type == "command_line":
            self.output_to_command_line(content)
        elif self.io_type == "message":
            self.output_to_message(content)
        else:
            raise ValueError("未知的输出方式")

    #====================================
    # 从命令行获取输入 
    # prompt 提示是给用户输入的提示词
    #====================================
    def get_input_from_command_line(self, prompt: str) -> str:
        # 从命令行获取输入 
        return input(prompt)

    #====================================
    # 从消息获取输入
    #====================================
    def get_input_from_message(self, prompt: str) -> str:
        # 这里假设从某个消息系统获取输入，具体实现需要根据实际情况编写
        # 例如从HTTP请求中获取数据

        content = {"content": "你是谁？"}
        #x = Msg(name="mytestname",role="user", content=content)
 
        return content.get("content")

    #====================================
    # 输出到命令行
    #====================================
    def output_to_command_line(self, content):
        print(content,flush=True)

    #====================================
    # 输出消息到用户
    #====================================
    def output_to_message(self, content):
        # 这里假设向某个消息系统输出，具体实现需要根据实际情况编写
        # 例如通过HTTP响应发送数据
        print(f"模拟的消息输出: {content}")

# 示例使用
'''
human_input = Human_Input_Output(input_method="command_line", output_method="command_line")
user_input = human_input.input("请输入命令行输入: ")
human_input.output(f"你输入的是: {user_input}")

human_input_message = Human_Input_Output(input_method="message", output_method="message")
user_input_message = human_input_message.input("请输入消息输入: ")
human_input_message.output(f"你输入的是: {user_input_message}")
'''
