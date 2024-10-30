import json
import asyncio
from abc import ABC,abstractmethod
from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Union,Sequence
import inspect
from collections import namedtuple
#import os

from ..logger import LOG_INFO,LOG_ERROR,LOG_WARNING,LOG_DEBUG,LOG_CRITICAL
from .model_response import ModelResponse
from ..msg.message import Msg
from .model_client import ModelClient
from ..tools.tool_manager import ToolManager
from ..utils.common import _convert_to_str
#from qwen_agent.llm import get_chat_model

#-----------------------------------
# 模型客户端，model wrapper 使用 。 
# 客户端的实际内容由 模型各个包装模块自己实现，
# 下面是模型客户端必须实现的方法，一共4个  
#  -- create_response_parser 创建 模型自己的响应解析方法 
#  -- get_message_from_response 利用解析方法将大模型的响应转换成 ModelResponse 类型 
#  -- cost 从client获取花销  
#  -- get_usage 从client获取使用情况
#-----------------------------------
class QwenChatWarpperClient(ModelClient):

    #千问系列默认api_type是 qwen    
    api_type:str = "qwen"

    """
    A client class must implement the following methods:
    - 这个client 对应的响应解码处理函数
    - cost 响应对应的cost
    - get_usage 获取使用情况包含下面5个内容
        - prompt_tokens
        - completion_tokens
        - total_tokens
        - cost
        - model

    This class is used to create a client that can be used by OpenAIWrapper.
    The response returned from create must adhere to the ModelClientResponseProtocol but can be extended however needed.
    The message_retrieval method must be implemented to return a list of str or a list of messages from the response.
    """
    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        organization: str = None,
        base_url: dict = None,
        stream: bool = False,
        generate_args: dict = None,
        tool_manager: ToolManager = None,
        **kwargs: Any,
    ) -> None:
        LOG_INFO("QwenChatWarpperClient::__init__-----------")
        self.model = model #保存模型
        self.generate_args = generate_args or {}
        #self.tools_manager = None 

        #===================================================
        # 导入qwen_agent, 本地qwen模型可以使用qwen_agent的接口进行处理
        #===================================================
        try:
            from qwen_agent.llm import get_chat_model
        except ImportError as e:
            raise ImportError(
                "Cannot find qwen_agent package, please install it by "
                "`pip install qwen_agent`",
            ) from e

        #===================================================
        # 加载openAI客户端
        #===================================================
        '''
        self.client = openai.OpenAI(
            api_key=api_key,
            organization=organization,
            base_url = base_url,
        )
        '''
        qwen_args = {}
        qwen_args.update(
            {
                "model": self.model,
                "model_server": base_url,
                "api_key": "EMPTY"
            },
        )
        # Use the OpenAI-compatible model service provided by DashScope:
        # 'model': 'qwen1.5-14b-chat',
        # 'model_server': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

        # Use the model service provided by Together.AI:
        # 'model': 'Qwen/Qwen1.5-14B-Chat',
        # 'model_server': 'https://api.together.xyz',  # api_base
        # 'api_key': os.getenv('TOGETHER_API_KEY'),

        # Use your own model service compatible with OpenAI API:
        # 'model': 'Qwen/Qwen1.5-72B-Chat',
        # 'model_server': 'http://localhost:8000/v1',  # api_base
        # 'api_key': 'EMPTY',

        # 千问的客户端
        self.client = get_chat_model(qwen_args)

        #===================================================
        # 将tool_manager 保存到当前w model client中
        #===================================================
        self.tools_manager = tool_manager

        # 流式标志 ，设置是否流式响应
        self.stream = stream   # 获取当前 是否是流式响应
        self.base_url = base_url # 访问的 url

        # Set the max length of OpenAI model
        if 'max_tokens' in kwargs:
            self.max_tokens = kwargs['max_tokens']  # 目前暂时这么设置，后续可以设置
        else:
            self.max_tokens = 4096  # 目前暂时这么设置，后续可以设置

    #---------------------------------------------------------
    # 将用户消息格式化成本模型支持的
    #---------------------------------------------------------
    def format(self, *args: Union[Msg, Sequence[Msg]]) -> List[dict]:
        messages = []
        #---------------------------------------------------
        #    遍历list中所有的消息 Msg   
        #         -- 普通用户消息 Msg 
        #         --大模型回的function call的 Msg 
        #         --调用完function call的结果，需要给大模型的 function call调用返回的结果的 Msg
        #---------------------------------------------------
        for arg in args:
            LOG_INFO("QwenChatWarpperClient::format=============== arg in args:",arg)
            if arg is None: # 为空则下一条
                continue
            if isinstance(arg, Msg): # 如果是 Msg
                if arg.url is not None:# 如果有 url
                    messages.append(self._format_msg_with_url(arg))
                else:
                    #---------------------------------------------------
                    #  如果是普通消息，将消息格式化成openai的格式
                    #---------------------------------------------------
                    if arg.role != "tool":
                        messages.append(
                            {
                                "role": arg.role,
                                "name": arg.name,
                                "content": _convert_to_str(arg.content),
                            },
                        )
                    else:
                        #---------------------------------------------------
                        #  下面是function call的响应消息，需要再完成  function call后转给大模型
                        #---------------------------------------------------
                        if arg.get("message"):
                            function_call_msg = arg.get("message")
                            if function_call_msg.get("raw"):
                                LOG_INFO("OpenAiChatWarpperClient::format:=============== message in Msg:",arg.get("message"))
                                fun_call_msgs  = function_call_msg.get("raw")
                                for fun_call_msg in fun_call_msgs:
                                    messages.append(fun_call_msg)

                        #---------------------------------------------------
                        # 下面是 function call 调用后的结果的消息
                         #---------------------------------------------------
                        elif arg.get("tool_responses"):
                            LOG_INFO("OpenAiChatWarpperClient::format:=============== tool_responses in Msg:",arg.get("tool_responses"))
                            openai_format_rsps = arg.get("tool_responses")   
                            for response in openai_format_rsps:
                                function_name = response["name"]  # 假设函数名称是已知的
                                function_content = response["content"]  # 将JSON字符串解析成字典
                                formatted_response = {
                                    'role': 'function',
                                    'name': function_name,
                                    'content': function_content,
                                }
                                # 将qwen格式的 调用了function call的结果 转换成 qwen需要的格式
                                messages.append(formatted_response)

                        else:
                            messages.append(arg)

                LOG_INFO("OpenAiChatWarpperClient::format=============== arg to message:",messages)

            elif isinstance(arg, list):
                messages.extend(self.format(*arg)) # 递归调用
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(arg)}.",
                )

        LOG_INFO("OpenAiChatWarpperClient::format=============== end! final messages:",messages)

        return messages
    
    # 将 function call 格式化成标准的openAI格式，后续处理tools call全部是这个格式
    def format_function_call_rsp(self, responses:List[dict]) -> List[dict]:

        #---------------------------------------------------
        # 1： 跟openAI返回的fucntion call 格式一样，
        #     只是message 中的内容不一样，这里是qwen的格式。
        #     需要转成 openAI的格式统一在外部处理
        #---------------------------------------------------
        '''
            [
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                    "name": "get_current_weather",
                    "arguments": "{\"location\": \"武汉\"}"
                    }
                },
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                    "name": "get_current_weather",
                    "arguments": "{\"location\": \"北京\"}"
                    }
                },
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                    "name": "get_current_weather",
                    "arguments": "{\"location\": \"深圳\"}"
                    }
                }
            ]

            下面这个是openAI的格式:
            {
                'id': 'chatcmpl-A0UthkSYtL6R1GyH8Ix0kHGUvDxar',
                'choices': [
                    {
                        'finish_reason': 'tool_calls',
                        'index': 0,
                        'logprobs': None,
                        'message': {
                            'content': None,
                            'role': 'assistant',
                            'function_call': None,
                            'tool_calls': [
                                {
                                    'id': 'call_ZtkKcdFIGTMet8upeKELwLVR',
                                    'function': {
                                    'arguments': '{"location": "武汉"}',
                                    'name': 'get_current_weather'
                                    },
                                    'type': 'function'
                                },
                                {
                                    'id': 'call_4IolGT147zkbWWvdufFfN1vA',
                                    'function': {
                                    'arguments': '{"location": "北京"}',
                                    'name': 'get_current_weather'
                                    },
                                    'type': 'function'
                                },
                                {
                                    'id': 'call_XSfvBPuBUhWb8Jo9Z2jJoRBK',
                                    'function': {
                                    'arguments': '{"location": "深圳"}',
                                    'name': 'get_current_weather'
                                    },
                                    'type': 'function'
                                }
                            ],
                            'refusal': None
                        }
                    }
                ],
                'created': 1724682133,
                'model': 'gpt-4o-2024-05-13',
                'object': 'chat.completion',
                'service_tier': None,
                'system_fingerprint': 'fp_c9aa9c0491',
                'usage': {
                    'completion_tokens': 61,
                    'prompt_tokens': 82,
                    'total_tokens': 143
                }
            }

        '''
        # 转换后的结果
        tool_calls = []

        # 生成新的格式
        for item in responses:
            # 不带function call的消息，直接跳过
            if item.get("function_call") is None:
               continue
            #call_id = f"call_{uuid.uuid4().hex}"  # 生成唯一的ID
            function_call = {
                #"id": call_id,
                "function": {
                    "arguments": item['function_call']['arguments'],
                    "name": item['function_call']['name']
                },
                "type": "function"
            }
            tool_calls.append(function_call)

        # 后续在啊tools调用的时候实际只需要tool_calls中的内容，而且这个必须在message中
        # raw 里面的内容主要是获取到tools调用结果后，一起送给大模型需要的格式。这个是qwen元素格式内容。
        # tool_calls是格式转成了openAI兼容格式,便于后面tools调用功能解析调用
        message = {
            "raw":responses,
            "tool_calls": tool_calls
        }
        # raw 里面存的是是 qwen原始是响应格式。后续需要用到
        #formatted_response =Msg(name="assistant",role="tool",content = "", message = message)
        LOG_INFO("QwenChatWarpperClient::format_function_call_rsp:----------- formatted_response:\n",message)
    
        return message

    #---------------------------
    # 响应解析器
    # qwen客户端的相应的解析器
    #---------------------------
    def _response_parser(self, params: Dict[str, Any]) -> ModelResponse:  
        pass

    # 同步应答
    # callback 这个回调函数是用来保存历史消息的，由于有stream类型的消息，所以需要回调函数来保存历史消息
    def _generate_response(
            self,
            messages: Sequence[dict],
            stream: Optional[bool] = None,
            stream_callback: Optional[Callable[[str], None]] = None,
            use_tools: bool = False,
            **kwargs: Any,
        ) -> ModelResponse:
        
        # step1: prepare keyword arguments
        kwargs = {**self.generate_args, **kwargs}
        LOG_INFO("QwenChatWarpperClient::_generate_response:----------- kwargs",kwargs)

        #-------------------------------------------------------------
        # 1： 首先将消息格式转成 本模型能够处理的格式，下面是openai的格式
        #-------------------------------------------------------------
        messages = self.format(messages)

        #-------------------------------------------------------------
        # 2： 参数检查，设置参数
        #-------------------------------------------------------------
        # step2: checking messages
        if not isinstance(messages, list):
            raise ValueError(
                "OpenAI `messages` field expected type `list`, "
                f"got `{type(messages)}` instead.",
            )
        
        '''
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for OpenAI API.",
            )
        '''

        # step3: forward to generate response
        if stream is None:
            stream = self.stream

        qwen_call_args = {}
        qwen_call_args.update(
            {
                "messages": messages,
                "stream": stream,
            },
        )

        #if stream:
        #    qwen_call_args["stream_options"] = {"include_usage": True}

        # 如果标识使用工具，并且有工具管理器,则获取工具描述
        if use_tools and self.tools_manager:
            tools = self.tools_manager.get_tool_descriptions(type = self.api_type)
            if len(tools) > 0:
                qwen_call_args["functions"] = tools

                #if use_tools and self.tools_manager:
                qwen_call_args["extra_generate_cfg"] = dict(
                # Note: set parallel_function_calls=True to enable parallel function calling
                #parallel_function_calls=True,  # Default: False
                parallel_function_calls=False,  # Default: False
                # Note: set function_choice='auto' to let the model decide whether to call a function or not
                # function_choice='auto',  # 'auto' is the default if function_choice is not set
                # Note: set function_choice='get_current_weather' to force the model to call this function
                # function_choice='get_current_weather',
                )
    
        #-------------------------------------------------------------
        # 3： 调用模型接口 同步
        #-------------------------------------------------------------
        LOG_INFO("QwenChatWarpperClient::_generate_response:------cccc----- qwen_call_args:\n",qwen_call_args)
        #response = self.client.chat.completions.create(**kwargs)
        responses = self.client.chat(**qwen_call_args)
        # 千万返回的是一个list，里面可能包含了多个消息
        LOG_INFO("QwenChatWarpperClient::_generate_response:------cccc----- response:\n",responses)

        if stream:
            #for chunk in responses:
            #    LOG_INFO("QwenChatWarpperClient::_a_generate_response: -------dddd--stream-- chunk:\n",chunk)
            #return None

            stream_messages = [] #流式消息内容可能是多个消息组成。内容肯是用户消息，也有可能是function call的消息
            has_function_call = False # 判断响应是否有funtion call
            text = ""

            # 遍历结果
            for chunk in responses:
                LOG_INFO("QwenChatWarpperClient::_generate_response: -------ccc-dddd--stream-- chunk:\n",chunk)
                #=======================================================
                # 流式消息,消息内容可能会有多个消息组成,按顺序一个消息一个消息的处理
                #=======================================================
                process_index = len(chunk) - 1 # 索引

                #--------------------------------------------------------
                #    如果索引和当前stream_messages的长度不匹配，则需要增加 stream_messages 的内容
                #    如果当前 stream_messages 消息长度小于等于 process_index ,
                #    流式消息每新增一条消息，stream_messages长度就需要一起增加
                #--------------------------------------------------------
                while len(stream_messages) <= process_index:
                    stream_messages.append(chunk[process_index])                

                # 如果是用户普通的消息
                if self._verify_text_content_in_qwen_stream_message_response(chunk[process_index]):
                    chunk_text = chunk[process_index]["content"]
                    text = chunk_text #逐个chunk组合起来
                    stream_messages[process_index]["content"] = chunk_text

                    #=======================================================
                    # 如果有回调，直接调用回调,将用户消息传递给用户
                    #=======================================================
                    if stream_callback is not None:
                        stream_callback(chunk_text)

                # 如果是function call的消息
                elif self._verify_function_call_in_qwen_stream_message_response(chunk[process_index]):
                    has_function_call = True
                    stream_messages[process_index]["function_call"] = chunk[process_index]["function_call"]

                else:
                    continue

            
            # 如果没有function call
            if has_function_call == False:
                LOG_INFO("QwenChatWarpperClient::_generate_response: -------ccc--stream-1-last chunk:\n",text)
                return ModelResponse(
                        text=text,
                    )
            else: # 如果是 function call的消息，则内容是 stream_message
                # 如果是tool call调用的消息，流式tool call如果多函数并发会多一个 "index": 0的字段，这个是用来定位流式位置的。
                # 而这个index内容在后续,函数调用结果一同传入qwen2.5的时候会报错，需要在接收完后将这个属性去掉
                LOG_INFO("QwenChatWarpperClient::_generate_response: -------ccc--stream-2-last function chunk:\n",stream_messages)
                new_stream_messages = self.format_function_call_rsp(stream_messages)
                LOG_INFO("QwenChatWarpperClient::_generate_response: -------ccc--stream-3-last new function chunk:\n",new_stream_messages)
                return ModelResponse(
                    text=new_stream_messages,
                    is_funcall_rsp=True,
                )  
        else:
            LOG_INFO("QwenChatWarpperClient::_generate_response:-------ccc222---- responses[0]:\n",responses[0])
            if len(responses) == 1:
                #判断当前响应是不是给用户的合法响应,这个响应只有一个内容
                if self._verify_text_content_in_qwen_message_response(responses[0]):
                    LOG_INFO("QwenChatWarpperClient::_generate_response:----------- rsp:\n",responses[0])
                    # return response
                    return ModelResponse(
                        text=responses[0]["content"],
                        #raw=responses[0],
                    )
                # 判断是否合法的 function call, 如果返回的是一个空的content，但是有function call的时候，这是一个function call消息
                elif self._verify_function_call_in_qwen_message_response(responses[0]):
                    # 将qwen格式的响应转换成标准的openai格式
                    LOG_INFO("QwenChatWarpperClient::_generate_response:-------ccc333-1---- org format response:\n",responses) 
                    new_rsp = self.format_function_call_rsp(responses)
                    LOG_INFO("QwenChatWarpperClient::_generate_response:-------ccc333-2---- new format response:\n",new_rsp) 
                
                    # 返回 response，放到 raw中,供后续使用， text取 choices[0]中的内容
                    return ModelResponse(
                        text=new_rsp,
                        #raw=responses,
                        is_funcall_rsp = True,
                    )
                
                else:
                    raise RuntimeError(
                        f"Invalid response from qwen API: {responses}",
                    )
            else:
                #----------------------------------------------------------------
                # 如果有多个消息，这种情况一般是
                #    第一个是说明(这个消息返回给用户)
                #    第二个消息是function call消息
                #----------------------------------------------------------------
                if self._verify_text_content_in_qwen_message_response(responses[0]):
                    LOG_INFO("QwenChatWarpperClient::_generate_response:----------- rsp:\n",responses[0])
                    #=======================================================
                    # 如果有回调，直接调用回调,将用户消息传递给用户
                    #=======================================================
                    if stream_callback is not None:
                        stream_callback(responses[0]["content"])

                if self._verify_function_call_in_qwen_message_response(responses[1]):
                    # 将qwen格式的响应转换成标准的openai格式

                    LOG_INFO("QwenChatWarpperClient::_generate_response:-------ccc333-3---- org format response:\n",responses) 
                    # 将qwen格式的响应转换成标准的openai格式, 这个是qwen2.5的格式文本方式，这个没有id
                    new_rsp = self.format_function_call_rsp(responses)
                    LOG_INFO("QwenChatWarpperClient::_generate_response:-------ccc333-4---- new format response:\n",new_rsp) 
                
                    # 返回 response，放到 raw中,供后续使用， text取 choices[0]中的内容
                    return ModelResponse(
                        text=new_rsp,
                        #raw=responses,
                        is_funcall_rsp = True,
                    )
                
                else:
                    raise RuntimeError(
                        f"Invalid response from qwen API: {responses}",
                    )                


    #==========================================================
    #
    # 异步应答，调用大模型接口，发送消息获取大模型响应
    # 
    #==========================================================
    async def _a_generate_response(
            self,
            messages: Sequence[dict],
            stream: Optional[bool] = None,
            stream_callback: Optional[Callable[[str], None]] = None,
            use_tools: bool = False,
            **kwargs: Any,
        ) -> ModelResponse:
        # 其他参数
        kwargs = {**self.generate_args, **kwargs}
        LOG_INFO("QwenChatWarpperClient::_a_generate_response:----------- kwargs",kwargs)

        #-------------------------------------------------------------
        # 1： 首先将消息格式转成 本模型能够处理的格式，下面是openai的格式
        #-------------------------------------------------------------
        messages = self.format(messages)

        #-------------------------------------------------------------
        # 2： 参数检查，设置参数
        #-------------------------------------------------------------
        # step2: checking messages
        if not isinstance(messages, list):
            raise ValueError(
                "OpenAI `messages` field expected type `list`, "
                f"got `{type(messages)}` instead.",
            )
        
        '''
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for OpenAI API.",
            )
        '''

        # step3: forward to generate response
        if stream is None:
            stream = self.stream

        qwen_call_args = {}
        qwen_call_args.update(
            {
                "messages": messages,
                "stream": stream,
            },
        )

        # 如果标识使用工具，并且有工具管理器,则获取工具描述
        if use_tools and self.tools_manager:
            qwen_call_args["functions"] = self.tools_manager.get_tool_descriptions(type = self.api_type)

            #if use_tools and self.tools_manager:
            qwen_call_args["extra_generate_cfg"] = dict(
            # Note: set parallel_function_calls=True to enable parallel function calling
            parallel_function_calls=True,  # Default: False
            #parallel_function_calls=False,  # Default: False
            # Note: set function_choice='auto' to let the model decide whether to call a function or not
            # function_choice='auto',  # 'auto' is the default if function_choice is not set
            # Note: set function_choice='get_current_weather' to force the model to call this function
            # function_choice='get_current_weather',
            )

        #-------------------------------------------------------------
        # 3： 调用模型接口 异步
        #-------------------------------------------------------------
        LOG_INFO("QwenChatWarpperClient::_a_generate_response:------cccc----- qwen_call_args:\n",qwen_call_args)
        responses = self.client.chat(**qwen_call_args)
        LOG_INFO("QwenChatWarpperClient::_a_generate_response:------cccc----- response:\n",responses)

        if stream:
            #for chunk in responses:
            #    LOG_INFO("QwenChatWarpperClient::_a_generate_response: -------dddd--stream-- chunk:\n",chunk)
            #return None

            #--------------------------------------------------------------
            #   首先取，第一个消息即可判断是 function call 还是普通消息
            #--------------------------------------------------------------
            first_chunk = next(responses)
            LOG_INFO("QwenChatWarpperClient::_a_generate_response:-------dddd-1-stream-- first_chunk:\n",first_chunk)

            stream_messages = [] #流式消息内容可能是多个消息组成。内容肯是用户消息，也有可能是function call的消息
            has_function_call = False # 判断响应是否有funtion call
            text = ""

            # 遍历结果
            for chunk in responses:
                LOG_INFO("QwenChatWarpperClient::_a_generate_response: -------ccc-dddd--stream-- chunk:\n",chunk)
                #=======================================================
                # 流式消息,消息内容可能会有多个消息组成,按顺序一个消息一个消息的处理
                #=======================================================
                process_index = len(chunk) - 1 # 索引

                #--------------------------------------------------------
                #    如果索引和当前stream_messages的长度不匹配，则需要增加 stream_messages 的内容
                #    如果当前 stream_messages 消息长度小于等于 process_index ,
                #    流式消息每新增一条消息，stream_messages长度就需要一起增加
                #--------------------------------------------------------
                while len(stream_messages) <= process_index:
                    stream_messages.append(chunk[process_index])                

                # 如果是用户普通的消息
                if self._verify_text_content_in_qwen_stream_message_response(chunk[process_index]):
                    chunk_text = chunk[process_index]["content"]
                    text = chunk_text #逐个chunk组合起来
                    stream_messages[process_index]["content"] = chunk_text

                    #=======================================================
                    # 如果有回调，直接调用回调,将用户消息传递给用户
                    #=======================================================
                    if stream_callback is not None:
                        stream_callback(chunk_text)

                # 如果是function call的消息
                elif self._verify_function_call_in_qwen_stream_message_response(chunk[process_index]):
                    has_function_call = True
                    stream_messages[process_index]["function_call"] = chunk[process_index]["function_call"]

                else:
                    continue

            
            # 如果没有function call
            if has_function_call == False:
                LOG_INFO("QwenChatWarpperClient::_a_generate_response: -------ccc--stream-1-last chunk:\n",text)
                return ModelResponse(
                        text=text,
                    )
            else: # 如果是 function call的消息，则内容是 stream_message
                # 如果是tool call调用的消息，流式tool call如果多函数并发会多一个 "index": 0的字段，这个是用来定位流式位置的。
                # 而这个index内容在后续,函数调用结果一同传入qwen2.5的时候会报错，需要在接收完后将这个属性去掉
                LOG_INFO("QwenChatWarpperClient::_a_generate_response: -------ccc--stream-2-last function chunk:\n",stream_messages)
                new_stream_messages = self.format_function_call_rsp(stream_messages)
                LOG_INFO("QwenChatWarpperClient::_a_generate_response: -------ccc--stream-3-last new function chunk:\n",new_stream_messages)
                return ModelResponse(
                    text=new_stream_messages,
                    is_funcall_rsp=True,
                )  
        else:
            LOG_INFO("QwenChatWarpperClient::_a_generate_response:-------ccc222---- responses[0]:\n",responses[0])
            # 这种情况一般用消息都只有一个消息内容
            if len(responses) == 1:
                #判断当前响应是不是给用户的合法响应,这个响应只有一个内容
                if self._verify_text_content_in_qwen_message_response(responses[0]):
                    LOG_INFO("QwenChatWarpperClient::_a_generate_response:----------- rsp:\n",responses[0])
                    return ModelResponse(
                        text=responses[0]["content"],
                        #raw=responses[0],
                    )
                # 判断是否合法的 function call, 如果返回的是一个空的content，但是有function call的时候，这是一个function call消息
                elif self._verify_function_call_in_qwen_message_response(responses[0]):
                    # 将qwen格式的响应转换成标准的openai格式
                    LOG_INFO("QwenChatWarpperClient::_a_generate_response:-------ccc333-1---- new format response:\n",responses) 
                    new_rsp = self.format_function_call_rsp(responses)
                    LOG_INFO("QwenChatWarpperClient::_a_generate_response:-------ccc333-2---- new format response:\n",new_rsp) 
                
                    # 返回 response，放到 raw中,供后续使用， text取 choices[0]中的内容
                    return ModelResponse(
                        text=new_rsp,
                        #raw=responses,
                        is_funcall_rsp = True,
                    )
                # 如果len不是1，这种情况一般可以认为是function call的消息
                else:
                    raise RuntimeError(
                        f"Invalid response from qwen API: {responses}",
                    )     
            else:
                #----------------------------------------------------------------
                # 如果有多个消息，这种情况一般是 function call.这个function call里面还有一部分是用户消息
                #    第一个是说明(这个消息返回给用户)
                #    第二个消息是function call消息
                #----------------------------------------------------------------
                if self._verify_text_content_in_qwen_message_response(responses[0]):
                    LOG_INFO("QwenChatWarpperClient::_a_generate_response:----------- rsp:\n",responses[0])
                    #=======================================================
                    # 如果有回调，直接调用回调,将用户消息传递给用户
                    #=======================================================
                    if stream_callback is not None:
                        stream_callback(responses[0]["content"])

                if self._verify_function_call_in_qwen_message_response(responses[1]):
                    # 将qwen格式的响应转换成标准的openai格式
                    LOG_INFO("QwenChatWarpperClient::_a_generate_response:-------ccc333-3---- org format response:\n",responses) 
                    new_rsp = self.format_function_call_rsp(responses)
                    LOG_INFO("QwenChatWarpperClient::_a_generate_response:-------ccc333-4---- new format response:\n",new_rsp) 
                
                    # 返回 response，放到 raw中,供后续使用， text取 choices[0]中的内容
                    return ModelResponse(
                        text=new_rsp,
                        #raw=responses,
                        is_funcall_rsp = True,
                    )
                # 如果len不是1，这种情况一般可以认为是function call的消息
                else:
                    raise RuntimeError(
                        f"Invalid response from qwen API: {responses}",
                    )  


    #---------------------------
    #  模型调用处理
    #  各个模型调用
    #---------------------------
    def __call__(
            self,
            messages: Sequence[dict],
            **kwargs: Any,
        ) -> ModelResponse:

        # 默认调用同步函数
        res = self._generate_response(
            messages=messages, 
            **kwargs)
    
        return res
            
    #---------------------------
    # 从响应中获取消息内容,将其转换成  ModelResponse 
    # 这是每个client必须实现的内容
    # 用创建的解析器对消息进行解析
    #---------------------------
    def get_message_from_response(
        self, response: Any
    ) -> Union[ModelResponse, List[ModelResponse]]:
        pass

    def _format_msg_with_url():
        pass

    #---------------------------
    #  获取 cost信息
    #
    #---------------------------
    def cost(self, response: ModelResponse) -> float:
        pass
    #---------------------------
    #  获取使用情况
    #
    #---------------------------
    def get_usage(self,response: ModelResponse) -> Dict:
        """Return usage summary of the response using RESPONSE_USAGE_KEYS."""
        pass

    #=======================================================
    #   检查 是否是function call的应答 在 qwen的响应消息里面
    #=======================================================
    def _verify_function_call_in_qwen_message_response(self,response: dict) -> bool:

        LOG_INFO("=_verify_function_call_in_qwen_message_response========0======response",response)
        #for message in response:
        if response.get("function_call", None) is None:
            LOG_INFO("=_verify_function_call_in_qwen_message_response========1======False")
            return False
        
        if response.get("content", None) is not None and response.get("content", None) != "":
            LOG_INFO("=_verify_function_call_in_qwen_message_response========1======False")
            return False
        
        return True

    #=======================================================
    #   检查 是否有内容 在 qwen的响应消息里面
    #=======================================================
    def _verify_text_content_in_qwen_message_response(self,response: dict) -> bool:
        if response.get("content", None) is None:
            LOG_INFO("=_verify_text_content_in_qwen_message_response========1======False")
            return False

        if response.get("content", None)  == "":
            LOG_INFO("=_verify_text_content_in_qwen_message_response========2======False")
            return False
        
        return True

    #=======================================================
    #   检查 是否是function call的应答 在 qwen的响应消息里面
    #=======================================================
    def _verify_function_call_in_qwen_stream_message_response(self,response: dict) -> bool:

        # stream模式下的 qwen 的 function call的消息中 ，content一定是 "" 空的，如果不为空，则不是function call
        if response.get("content", None) != "":
            LOG_INFO("=_verify_function_call_in_qwen_stream_message_response========1======False")
            return False

        # function_call 一定是存在的，并且有内容
        if response.get("function_call", None) is None:
            LOG_INFO("=_verify_function_call_in_qwen_stream_message_response========2======False")
            return False
        
        return True
    
    #==========================================================
    #  检查 是否有内容  qwen流式传输时候的消息内容处理，流式输出和非流式的有区别
    #==========================================================
    def _verify_text_content_in_qwen_stream_message_response(self,response: dict) -> bool:

        # stream模式下的 qwen 的 普通消息中 ，content一定不是空的，如果为空，则不是普通消息
        if response.get("content", None) == "":
            LOG_INFO("=_verify_text_content_in_qwen_stream_message_response========1======False")
            return False

        # stream模式下的 qwen 的 普通消息中, function_call 一定是空的，如果不为空，则不是普通消息
        if response.get("function_call", None) is not None:
            LOG_INFO("=_verify_text_content_in_qwen_stream_message_response========2======False")
            return False

        return True