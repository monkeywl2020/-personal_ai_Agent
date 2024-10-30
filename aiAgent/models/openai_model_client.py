import json
import asyncio
from abc import ABC,abstractmethod
from typing import Any, Callable, Dict, List, Optional, Generator, Tuple, Union,Sequence
import inspect

from ..logger import LOG_INFO,LOG_ERROR,LOG_WARNING,LOG_DEBUG,LOG_CRITICAL
from .model_response import ModelResponse
from ..msg.message import Msg
from .model_client import ModelClient
from ..tools.tool_manager import ToolManager
from ..utils.common import _convert_to_str
#-----------------------------------
# 模型客户端，model wrapper 使用 。 
# 客户端的实际内容由 模型各个包装模块自己实现，
# 下面是模型客户端必须实现的方法，一共4个  
#  -- create_response_parser 创建 模型自己的响应解析方法 
#  -- get_message_from_response 利用解析方法将大模型的响应转换成 ModelResponse 类型 
#  -- cost 从client获取花销  
#  -- get_usage 从client获取使用情况
#-----------------------------------
class OpenAiChatWarpperClient(ModelClient):

    #千问系列默认api_type是 qwen    
    api_type:str = "openai"

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
        LOG_INFO("OpenAiChatWarpperClient::__init__-----------")
        self.model = model #保存模型
        self.generate_args = generate_args or {}

        #===================================================
        # 导入openai, 本地qwen模型可以使用openAI的接口进行处理
        #===================================================
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "Cannot find openai package, please install it by "
                "`pip install openai`",
            ) from e

        #===================================================
        # 加载openAI客户端
        #===================================================
        #self.client = openai.AsyncOpenAI(
        self.client = openai.OpenAI(
            api_key=api_key,
            organization=organization,
            base_url = base_url,
        )

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
            LOG_INFO("OpenAiChatWarpperClient::format:=============== arg in args:",arg)
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
                        #  如果是工具调用相关的消息，将消息格式化成openai的格式
                        #---------------------------------------------------
                        if arg.get("message"):
                            LOG_INFO("OpenAiChatWarpperClient::format:=============== message in Msg:",arg.get("message"))
                            messages.append(arg.get("message"))

                        # 下面是工具调用结果的消息
                        elif arg.get("tool_responses"):
                            LOG_INFO("OpenAiChatWarpperClient::format:=============== tool_responses in Msg:",arg.get("tool_responses"))
                            for tool_response in arg.get("tool_responses"):
                                messages.append(tool_response)
                            #messages.append()

                        else:
                            messages.append(arg)
                LOG_INFO("OpenAiChatWarpperClient::format:=============== arg to message:",messages)

            elif isinstance(arg, list):
                messages.extend(self.format(*arg)) # 递归调用
            else:
                raise TypeError(
                    f"The input should be a Msg object or a list "
                    f"of Msg objects, got {type(arg)}.",
                )

        LOG_INFO("OpenAiChatWarpperClient::format:=============== end! final messages:",messages)
        return messages
    
    #---------------------------
    # 响应解析器
    # qwen客户端的响应的解析器
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
        LOG_INFO("OpenAiChatWarpperClient::_generate_response:---------11-- kwargs",kwargs)

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

        openAIkwargs ={}
        openAIkwargs.update(
            {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "max_tokens": self.max_tokens
            },
        )

        if stream:
            openAIkwargs["stream_options"] = {"include_usage": True}

        # 如果标识使用工具，并且有工具管理器,则获取工具描述
        if use_tools and self.tools_manager:
            tools = self.tools_manager.get_tool_descriptions(type = "openai")
            if len(tools) > 0:
                openAIkwargs["tools"] = tools

        #-------------------------------------------------------------
        # 3： 调用模型接口 同步
        #-------------------------------------------------------------
        LOG_INFO("OpenAiChatWarpperClient::_generate_response:-------cccc---- openAIkwargs:\n",openAIkwargs)
        response = self.client.chat.completions.create(**openAIkwargs)
        LOG_INFO("OpenAiChatWarpperClient::_generate_response:-------cccc---- response:\n",response)

        if stream:
            #for chunk in response:
            #    chunk = chunk.model_dump()
            #    LOG_INFO("Qwen2_5_OpenAiChatWarpperClient::_generate_response: -------dddd--stream-- chunk:\n",chunk)
            #return None
            #=======================================================
            #   遍历流式响应，获取完整的响应,调用用户侧的回调函数，
            #   将流式信息传递给用户
            #=======================================================
            first_chunk = next(response) # 默认第一个消息只携带  role 内容 
            first_chunk = first_chunk.model_dump()
            stream_message = first_chunk["choices"][0]["delta"]
            stream_message["tool_calls"] = []
            stream_message["content"] = ""
            text = ""

            # 遍历llm返回的流式响应
            for chunk in response:
                chunk = chunk.model_dump()
                LOG_INFO("OpenAiChatWarpperClient::_generate_response: -------ccc-dddd--stream-- chunk:\n",chunk)

                # 如果第一个消息是 null message，则取下一个消息，空消息丢弃
                if self._verify_text_content_in_openai_stream_message_response(chunk):
                    chunk_text = chunk["choices"][0]["delta"]["content"]
                    text += chunk_text #逐个chunk组合起来

                    #=======================================================
                    # 如果有回调，直接调用回调,将用户消息传递给用户
                    #=======================================================
                    if stream_callback is not None:
                        stream_callback(chunk_text)

                # 如果是function call的消息
                elif self._verify_function_call_in_openai_stream_message_response(chunk):
                    # 进入这里表示有 function call内容。
                    # 获取 function call 的index 
                    function_index = chunk["choices"][0]["delta"]["tool_calls"][0]["index"]

                    # 如果 tool_calls 列表的大小小于index，那就要增加tool_calls的列表内容
                    while len(stream_message["tool_calls"]) <= function_index:
                        stream_message["tool_calls"].append(chunk["choices"][0]["delta"]["tool_calls"][0])

                    if chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "":
                        continue
                    else:
                        # tool_calls 列表添加了index指定的 元素后，后面
                        stream_message["tool_calls"][function_index]["function"]["arguments"] += chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] 

                else:
                    continue

            #=======================================================
            # 最后完整的内容报文  
            #=======================================================     
            stream_message["content"] =  text # 将组合好的文本内容放到content中去
            # 如果是普通用户消息，text内容就是组合文本
            if len(stream_message["tool_calls"]) == 0:# 用这个判断是因为 openAI格式的tools call内容都在一个dict中
                LOG_INFO("OpenAiChatWarpperClient::_generate_response: -------ccc--stream-1-last chunk:\n",text)
                return ModelResponse(
                        text=text,
                    )
            else: # 如果是 function call的消息，则内容是 stream_message
               # 如果是tool call调用的消息，流式tool call如果多函数并发会多一个 "index": 0的字段，这个是用来定位流式位置的。
                # 而这个index内容在后续,函数调用结果一同传入qwen2.5的时候会报错，需要在接收完后将这个属性去掉
                LOG_INFO("OpenAiChatWarpperClient::_generate_response: -------ccc--stream-2-last function chunk:\n",stream_message)
                return ModelResponse(
                    text=stream_message,
                    is_funcall_rsp=True,
                )  
                        
        else:
            response = response.model_dump()
            LOG_INFO("OpenAiChatWarpperClient::_generate_response:-------ccc222---- response:\n",response)
            #self._save_model_invocation_and_update_monitor(
            #    kwargs,
            #    response,
            #)            
            if self._verify_text_content_in_openai_message_response(response):
                return ModelResponse(
                    text=response["choices"][0]["message"]["content"],
                    #raw=response,
                )
            # 如果是tools的function call响应。
            elif self._verify_function_call_in_openai_message_response(response):
                # 返回 response，放到 raw中,供后续使用， text取 choices[0]中的内容
                return ModelResponse(
                    text=response["choices"][0]["message"],
                    #raw=response,
                    is_funcall_rsp = True,
                )
            else:
                raise RuntimeError(
                    f"Invalid response from OpenAI API: {response}",
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
        LOG_INFO("OpenAiChatWarpperClient::_a_generate_response-----------")
        # 其他参数
        kwargs = {**self.generate_args, **kwargs}

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
        
        # 所有消息必须含有  role 和 content 关键字
        if not all("role" in msg and "content" in msg for msg in messages):
            raise ValueError(
                "Each message in the 'messages' list must contain a 'role' "
                "and 'content' key for OpenAI API.",
            )

        # step3: forward to generate response
        if stream is None: #如果没有传入的参数
            stream = self.stream #直接获取 初始化 的stream

        openAIkwargs={}
        openAIkwargs.update(
            {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "max_tokens": self.max_tokens
            },
        )

        #如果有 stream 为 true
        if stream:
            openAIkwargs["stream_options"] = {"include_usage": True} # 获取 usage

        # 如果标识使用工具，并且有工具管理器,则获取工具描述
        if use_tools and self.tools_manager:
            openAIkwargs["tools"] = self.tools_manager.get_tool_descriptions(type = "openai")

        #-------------------------------------------------------------
        # 3： 调用模型接口 异步
        #-------------------------------------------------------------
        LOG_INFO("OpenAiChatWarpperClient::_a_generate_response:-------cccc---- openAIkwargs:\n",openAIkwargs)
        response = self.client.chat.completions.create(**openAIkwargs)
        LOG_INFO("OpenAiChatWarpperClient::_a_generate_response:-------cccc---- response:\n",response)

        if stream:
            #for chunk in response:
            #    chunk = chunk.model_dump()
            #    LOG_INFO("Qwen2_5_OpenAiChatWarpperClient::_generate_response: -------dddd--stream-- chunk:\n",chunk)
            #return None
            #=======================================================
            #   遍历流式响应，获取完整的响应,调用用户侧的回调函数，
            #   将流式信息传递给用户
            #=======================================================
            first_chunk = next(response) # 默认第一个消息只携带  role 内容 
            first_chunk = first_chunk.model_dump()
            stream_message = first_chunk["choices"][0]["delta"]
            stream_message["tool_calls"] = []
            stream_message["content"] = ""
            text = ""

            # 遍历llm返回的流式响应
            for chunk in response:
                chunk = chunk.model_dump()
                LOG_INFO("OpenAiChatWarpperClient::_a_generate_response: -------ccc-dddd--stream-- chunk:\n",chunk)

                # 如果第一个消息是 null message，则取下一个消息，空消息丢弃
                if self._verify_text_content_in_openai_stream_message_response(chunk):
                    chunk_text = chunk["choices"][0]["delta"]["content"]
                    text += chunk_text #逐个chunk组合起来

                    #=======================================================
                    # 如果有回调，直接调用回调,将用户消息传递给用户
                    #=======================================================
                    if stream_callback is not None:
                        stream_callback(chunk_text)

                # 如果是function call的消息
                elif self._verify_function_call_in_openai_stream_message_response(chunk):
                    # 进入这里表示有 function call内容。
                    # 获取 function call 的index 
                    function_index = chunk["choices"][0]["delta"]["tool_calls"][0]["index"]

                    # 如果 tool_calls 列表的大小小于index，那就要增加tool_calls的列表内容
                    while len(stream_message["tool_calls"]) <= function_index:
                        stream_message["tool_calls"].append(chunk["choices"][0]["delta"]["tool_calls"][0])

                    if chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == "":
                        continue
                    else:
                        # tool_calls 列表添加了index指定的 元素后，后面
                        stream_message["tool_calls"][function_index]["function"]["arguments"] += chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] 

                else:
                    continue

            #=======================================================
            # 最后完整的内容报文  
            #=======================================================     
            stream_message["content"] =  text # 将组合好的文本内容放到content中去
            # 如果是普通用户消息，text内容就是组合文本
            if len(stream_message["tool_calls"]) == 0:
                LOG_INFO("OpenAiChatWarpperClient::_a_generate_response: -------ccc--stream-1-last chunk:\n",text)
                return ModelResponse(
                        text=text,
                    )
            else: # 如果是 function call的消息，则内容是 stream_message
               # 如果是tool call调用的消息，流式tool call如果多函数并发会多一个 "index": 0的字段，这个是用来定位流式位置的。
                # 而这个index内容在后续,函数调用结果一同传入qwen2.5的时候会报错，需要在接收完后将这个属性去掉
                LOG_INFO("OpenAiChatWarpperClient::_a_generate_response: -------ccc--stream-2-last function chunk:\n",stream_message)
                return ModelResponse(
                    text=stream_message,
                    is_funcall_rsp=True,
                )     
        
        else:
            response = response.model_dump()
            LOG_INFO("OpenAiChatWarpperClient::_a_generate_response:-------cccc222---- response:model_dump():\n",response)
            #self._save_model_invocation_and_update_monitor(
            #    kwargs,
            #    response,
            #)
            if self._verify_text_content_in_openai_message_response(response):
                # return response
                return ModelResponse(
                    text=response["choices"][0]["message"]["content"],
                    #raw=response,
                )
            # 如果是tools的function call响应。
            elif self._verify_function_call_in_openai_message_response(response):
                # 返回 response，放到 raw中,供后续使用， text取 choices[0]中的内容
                return ModelResponse(
                    text=response["choices"][0]["message"],
                    #raw=response,
                    is_funcall_rsp = True,
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
    def _verify_function_call_in_openai_message_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========1======False")
            return False

        if response["choices"][0].get("message", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========2======False")
            return False
        
        if response["choices"][0]["message"].get("tool_calls", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========3======False")
            return False
        
        return True
        

    #=======================================================
    #   检查 是否有内容 在 qwen的响应消息里面
    #=======================================================
    def _verify_text_content_in_openai_message_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========1======False")
            return False

        if response["choices"][0].get("message", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========2======False")
            return False

        if response["choices"][0]["message"].get("content", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========3======False")
            return False

        return True

    #=======================================================
    #   检测流式响应是否是空消息，
    #    多function call的情况下，第一个消息可能是空消息，content 为null，同时tool_calls也为null
    #=======================================================
    def _verify_null_msg_in_openai_stream_response(self,response: dict) -> bool:
        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========0======False")
            return False

        if response["choices"][0].get("delta", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========3======False")
            return False
        
        # stream模式下的 openai 的消息中，空消息的 content一定是 null 如果不为空，则不是 null stream msg
        if response["choices"][0]["delta"].get("content",None) is not None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========4======False")
            return False
                
        # stream模式下的 openai 的消息中，空消息的 tool_calls 一定是 null 如果不为空，则不是 null stream msg
        if response["choices"][0]["delta"].get("tool_calls", None) is not None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========5======False")
            return False
        
        return True
    
    #=======================================================
    #   检查 是否是function call的应答 在 qwen的响应消息里面
    #=======================================================
    def _verify_function_call_in_openai_stream_message_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========0======False")
            return False
        
        # stream模式下的 openai的 function call的消息中 ，content一定是 null，空的，如果不为空，则不是function call
        if response["choices"][0]["delta"].get("content",None) is not None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========4======False")
            return False
                
        if response["choices"][0]["delta"].get("tool_calls", None) is None:
            LOG_INFO("=_verify_text_content_in_openAI_message_response========45======False")
            return False
        
        return True

    #==========================================================
    #  检查 是否有内容  qwen流式传输时候的消息内容处理，流式输出和非流式的有区别
    #==========================================================
    def _verify_text_content_in_openai_stream_message_response(self,response: dict) -> bool:

        if len(response.get("choices", [])) == 0:
            LOG_INFO("=_verify_text_content_in_openai_stream_message_response========1======False")
            return False

        if response["choices"][0].get("delta", None) is None:
            LOG_INFO("=_verify_text_content_in_openai_stream_message_response========2======False")
            return False

        if response["choices"][0]["delta"].get("content", None) is None:
            LOG_INFO("=_verify_text_content_in_openai_stream_message_response========3======False")
            return False

        return True