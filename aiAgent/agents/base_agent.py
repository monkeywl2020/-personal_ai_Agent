# -*- coding: utf-8 -*-
""" Base class for Agent """
import copy
import uuid
import inspect
import asyncio
import functools
import time
import json
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Generator, Union,DefaultDict,Sequence
from collections import defaultdict
from termcolor import colored

from ..logger import LOG_INFO,LOG_ERROR,LOG_WARNING,LOG_DEBUG,LOG_CRITICAL
from .operator import Operator
from ..memory.memorybase import MemoryBase
from ..memory.memory_temporary import TemporaryMemory
from ..knowledge import Knowledge
from ..msg.message import Msg
from ..models.model_wrapper import ModelChatWrapper
from ..models.model_response import ModelResponse
from ..io.human_io import HumanIOHandler
from ..tools.tool_base import BaseTool, AgentTool
from ..tools.tool_manager import ToolManager

# BaseAgent继承自Operator
class BaseAgent(Operator):

    # 类级别的常量属性
    MAX_NUM_AUTO_REPLY = 50  # 连续自动回复的最大次数 暂定
    DEFAULT_CONFIG = False  # False or dict, the default config for llm inference   

    # 类型注解，用于声明属性的类型
    #---------------------------------------------------------
    # 1：本agent id 全球唯一id uuid.uuid4().hex,创建的时候自动生成
    #---------------------------------------------------------
    _id: str

    #---------------------------------------------------------
    # 2：本agent名称
    #---------------------------------------------------------
    _name: str

    #---------------------------------------------------------
    # 3：本agent系统提示词 sys_prompt 既可以是 str或者list 类型，也可以是 None
    #---------------------------------------------------------
    _sys_prompt: Optional[Union[str, List]]

    #---------------------------------------------------------
    # 4：本agent的 模型配置列表和其他模型相关的参数
    '''
    config_list=[
        {
            "model": "gpt-4",
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "api_type": "azure",
            "base_url": os.environ.get("AZURE_OPENAI_API_BASE"),
            "api_version": "2024-02-01",
        },
        {
            "model": "gpt-3.5-turbo",
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "api_type": "openai",
            "base_url": "https://api.openai.com/v1",
        },
        {
            "model": "llama-7B",
            "base_url": "http://127.0.0.1:8080",
        },
        {
            "model": "/work/wl/.cache/modelscope/hub/qwen/Qwen2-72B-Instruct-GPTQ-Int4",
            "base_url": "http://172.21.30.230:8980/v1",
            "api_type": "qwen",
            "api_key": "NotRequired",
            "stream": True
        }
    ]   
    llm_config_example = 
    {
        "config_list": config_list, # 模型列表
    }    
    '''
    #  参数类型 可以是字典 or False or None
    #---------------------------------------------------------
    llm_config: Optional[Union[Dict, Literal[False]]]

    #---------------------------------------------------------
    # 5：本agent 的简短描述。其他agent使用此描述来决定何时呼叫此agent。
    # （用户没赋值，就使用默认值: sys_prompt)
    #---------------------------------------------------------
    _description: Optional[str]

    #---------------------------------------------------------
    # 6： human_input_mode 用户输入模式。ALWAYS, TERMINATE, NEVER 只有这3中输入值
    #    用户输入模式: ALWAYS    用户一直干预,等待用户输入，且只有当用户输入 exit 或者 is_stop_msg 为True时且没有人工输入，则停止对话
    #                 TERMINATE 根据用户设置的条件干预，当  max_num_auto_reply 达到最大连续自动回复数次数时停止
    #                 NEVER     不进行人工干预, 只有 is_stop_msg 为true 或者 max_num_auto_reply 达到最大值才停止对话
    #---------------------------------------------------------
    human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"]

    #---------------------------------------------------------
    # 7： 用户输入类型，目前支持命令行输入，或者 消息输入例如：{"content": "你是谁？"}
    #     "command_line": 命令行输入  "message": 消息输入  
    #---------------------------------------------------------

    human_io_type: Optional[str] 
    #---------------------------------------------------------
    # 8： 日志记录器，记录本agent的日志信息
    #      挂载 HumanIOHandler 实例
    #---------------------------------------------------------
    human_io_handler: Optional[HumanIOHandler]

    #---------------------------------------------------------
    # 9： is_stop_msg （函数）：以字典形式接收消息的函数  
    #     函数的参数是 字典形式 {}
    #     返回结果是 True 和 False 
    #              ：True表示停止消息  
    #              ：False 表示不停止
    #     用户输入模式: 
    #---------------------------------------------------------
    is_stop_msg: Optional[Callable[[Dict], bool]] 

    #---------------------------------------------------------
    # 10：  max_num_auto_reply 最大连续自动回复数次数  
    #     设置了这个参数后，agent回复达到这个次数自动停止
    #---------------------------------------------------------
    _max_num_auto_reply: int

    # 记录agent对某个agent当前自动回复的次数
    _auto_reply_counter: DefaultDict[Any, int] 
    # 记录agent对某个最大自动回复的次数，默认是50
    _max_num_auto_reply_dict: DefaultDict[Any, int] 

    #---------------------------------------------------------
    # 13： use_memory 是否开启记忆
    #     设置了这个参数后agent会记录对话信息
    #---------------------------------------------------------
    use_memory: bool

    #---------------------------------------------------------
    # 14： memory 存储历史记录和上下文
    #     设置了use_memory 才有效，MemoryBase 可以为内存，redis，mysql，minvuls
    #---------------------------------------------------------
    memory: Optional[MemoryBase]

    #---------------------------------------------------------
    # 16： agent_objs 是多个注册到这个agent的实例的集合
    #     是一个列表List[BaseAgent],[BaseAgent1,BaseAgent2,BaseAgent3.....]
    #---------------------------------------------------------
    agent_lists: Optional[List["BaseAgent"]] 

    #---------------------------------------------------------
    # 17： knowledge_list 多个知识对象知识对象，知识对象里面使用RAG
    #     是一个列表List[BaseAgent],[BaseAgent1,BaseAgent2,BaseAgent3.....]
    #---------------------------------------------------------
    knowledge_list: Optional[List[Knowledge]] 

    #---------------------------------------------------------
    # 18： user_data 用户数据,格式自定义
    #      
    #---------------------------------------------------------
    user_data: Optional[Any]

    #---------------------------------------------------------
    # 19： 日志记录器，记录本agent的日志信息
    #      
    #---------------------------------------------------------
    log_recorder: Optional[Any] 

    #---------------------------------------------------------
    # 20： agent 切换策略,字典或者 auto or @
    #      
    #---------------------------------------------------------
    agent_switching_strategy: Optional[Union[Dict, Literal["Auto","@"]]]

    #---------------------------------------------------------
    # 21： agent 生成响应的钩子函数
    #      
    #---------------------------------------------------------   
    _reply_func_list : List[Callable]
    #---------------------------------------------------------
    # 22： agent 各个阶段的钩子函数
    #      
    #---------------------------------------------------------    
    hook_lists: Optional[Dict[str, List[Callable]]] 

    #---------------------------------------------------------
    # 23： 人类操作的历史记录
    #      
    #--------------------------------------------------------- 
    _human_input: List[Any]

    #---------------------------------------------------------
    # 15： Tools 工具集,可以挂载多个工具集和
    #     所有tools通过AgentTool转换成AgentTool实例,后续挂载带agent上
    #      
    #---------------------------------------------------------
    tool_manager: Optional[ToolManager]

    #---------------------------------------------------------
    # 24： 是否对外 隐藏 tool call调用
    #      默认不隐藏，也就是tools call不会再 llm响应后立刻调用，
    #      而是将结果返回给用户，用户再决定是否调用。如果是Ture的话
    #      则会在llm响应确定是tools call的时候会自行调用tools call
    #--------------------------------------------------------- 
    _hide_toolscall: bool 

    #---------------------------------------------------------
    # 25： tool call自调用最大次数，如果超过这个次数就不再调用 
    #      防止调用过多影响响应的速度，默认5次
    #      
    #    
    #--------------------------------------------------------- 
    _max_num_tools_call: int
    
    #初始化
    def __init__(
        self,
        name: str,
        sys_prompt: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        description: Optional[str] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        is_stop_msg: Optional[Callable[[Dict], bool]] = None,
        max_num_auto_reply: Optional[int] = None,
        use_memory: bool = True,
        memoryType:  Optional[str] = None,
        agent_lists: Optional[List["BaseAgent"]] = None,
        knowledge_list: Optional[List[Knowledge]] = None,
        userdata:Optional[Any] = None,
        log_recorder:Optional[Any] = None,
        hide_toolscall: Optional[bool] = False,
        **kwargs: Any,
    ):
        LOG_INFO("BaseAgent-----------__init__:",name)
        self._id = self.__class__.generate_agent_id()        
        
        # 记录名称,名字必须有
        self._name = name

        #设置系统消息
        self._sys_prompt = sys_prompt # [{"content": sys_prompt, "role": "system"}]
        
        # 默认这个agent的简要描述如果没有填的话，直接使用 sys_prompt
        self._description = description if description is not None else sys_prompt
        
        #用户干预类型
        self.human_input_mode = human_input_mode

        # 获取当前停止函数，如果没定义则用默认的函数（如果消息中携带了 teminate则是停止）
        # 判断当前消息里面是否有 有停止标志（如果没有判断函数，默认使用判断content里面内容是否含TERMINATE ）
        self.is_stop_msg = (
            is_stop_msg
            if is_stop_msg is not None
            else (lambda x: x.get("content") == "TERMINATE")#如果消息里面有TERMINATE 则表示停止
        )

        #最大自动应答次数
        self._max_num_auto_reply = (
            max_num_auto_reply if max_num_auto_reply is not None else self.MAX_NUM_AUTO_REPLY
        )

        #对某个agent当前自动回复的次数
        self._auto_reply_counter = defaultdict(int)
        #对某个最大自动回复的次数，默认是50
        self._max_num_auto_reply_dict = defaultdict(self.get_max_num_auto_reply)

        #是否使用记忆
        self.use_memory = use_memory
        # 开启记忆后，会有持续的历史记录和上下文，memory是一个实例挂过来的
        if use_memory:
            if memoryType is None:
                self.memory = TemporaryMemory() # 挂内存实例，后续可以根据内存内容挂，暂时测试先这样用
            else: # 这里应该是挂其他类型的memory，目前测试先用临时内存
                self.memory = TemporaryMemory() # 挂内存实例，后续可以根据内存内容挂，暂时测试先这样用
        else:
            self.memory = None # 如果没有开启记忆就不用挂
    
        #设置 这个agent关联的 agent列表，多个agent是用来做任务协作的
        self.agent_lists = agent_lists

        #挂载知识库
        self.knowledge_list = knowledge_list
 
        #挂载用户自定义数据，这部分内容是存储用户自定义数据使用的，在agent使用过程中可以用这个内容存储用户的数据
        self.user_data = userdata 
        
        #日志记录
        self.log_recorder = log_recorder

        #默认 @ 符号 agent名字切换
        self.agent_switching_strategy = "@"

        #默认人类输入方式是命令行输入
        self.human_io_type = kwargs.get('human_io_type', "command_line")

        #挂载输入输出处理器
        self.human_io_handler = HumanIOHandler(io_type=self.human_io_type) #挂载输入输出处理器

        # 记录人类输入的所有内容，作为人类输入的操作记录
        self._human_input = []

        # 初始化 _reply_func_list 为实例属性
        self._reply_func_list = []

        # 注册钩子函数，这写钩子函数是挂载在各个消息从处理位置上的，初始化默认全部为空
        self.hook_lists = {
            "process_received_message": [],
            "process_all_messages_before_reply": [],
            "process_message_before_send": [],
        }
        
        #是否隐藏tools call调用，确定 reply的流程
        self._hide_toolscall = hide_toolscall
        LOG_INFO("BaseAgent-----------2222__init__:hide_toolscall", hide_toolscall)

        # 挂载工具管理器,这个是用来管理工具的，挂载到agent上,tools通过 注册将tools挂载到这个管理器上
        self.tool_manager = ToolManager() #挂载工具管理器

        # 初始化工具调用最大次数
        self._max_num_tools_call = 5

        # 拷贝 llm_config
        if isinstance(llm_config, dict):
            try:
                llm_config = copy.deepcopy(llm_config)
            except TypeError as e:
                raise TypeError(
                    "Please implement __deepcopy__ method for each value class in llm_config to support deepcopy."
                ) from e
        # 检查 llm_config 配置并加载 llm 包装器，初始化的时候根据配置加载好
        self._validate_llm_config_and_load(llm_config, self.tool_manager)

        #-----------------------------------------------------------------------
        #下面这 函数是注册到 发送响应处理时候的函数，也就是agent生成响应要做的处理。
        # 顺序如下：
        #       1：处理 humam 输入信息，停止或人类干预。
        #       2：处理 发送出去的消息 
        #       3：处理tools处理信息。可能要调用大模型 
        #-----------------------------------------------------------------------
        # 注册agent的处理函数, 下面 默认  对 agent和 none 两种进行触发处理
        #self.register_reply([None], BaseAgent.generate_tool_calls_reply,ignore_sync_in_async_chat=True) # 异步调用的时候忽略同步
        #self.register_reply([None], BaseAgent.a_generate_tool_calls_reply, ignore_async_in_sync_chat=True)
        

        LOG_INFO("BaseAgent-----------1111__init__:\n_reply_func_list:", self.name, self._reply_func_list)

        self.register_reply([None], BaseAgent.generate_llm_reply,ignore_sync_in_async_chat=True)# 同步函数, 异步调用的时候忽略同步
        self.register_reply([None], BaseAgent.a_generate_llm_reply, ignore_async_in_sync_chat=True)# 异步函数
        
        # 注册tools调用处理函数，这个部分是在调用llm之前的。如果将tools调用对用户屏蔽，会导致一个问题，tools调用不可控，同时用户等待时间长（多个tools调用的时候）
        self.register_reply([None], BaseAgent.generate_tool_calls_reply,ignore_sync_in_async_chat=True) # 异步调用的时候忽略同步
        self.register_reply([None], BaseAgent.a_generate_tool_calls_reply, ignore_async_in_sync_chat=True)

        #最后insert 处理的时候第一个处理，所有的 agent的消息都是先走人工干预处理开始
        self.register_reply([None], BaseAgent.check_termination_and_human_reply,ignore_sync_in_async_chat=True) # 异步调用的时候忽略同步
        self.register_reply([None], BaseAgent.a_check_termination_and_human_reply, ignore_async_in_sync_chat=True)

        LOG_INFO("BaseAgent-----------2222__init__:\n_reply_func_list", self.name, self._reply_func_list)

        # 注册hook，这个钩子是挂载在 process_received_message 接收到消息的时候的        
        self.register_hook(hookable_method="process_received_message", hook = self._print_rcv_message)
        self.register_hook(hookable_method="process_received_message", hook = self._preprocess_rcv_message)
        self.register_hook(hookable_method="process_message_before_send", hook = self._process_send_message)
        LOG_INFO("BaseAgent-----------__init__:\nhook_lists",self.hook_lists)

    #--------------------------------
    #类方法，创建的时候自动生成 id
    #--------------------------------
    @classmethod
    def generate_agent_id(cls) -> str:
        """Generate the agent_id of this agent instance"""
        # TODO: change cls.__name__ into a global unique agent_type
        return uuid.uuid4().hex

    @property
    def name(self) -> str:
        """获取本agent的名称."""
        return self._name

    @property
    def sysprompt(self) -> str:
        """获取本agent的系统提示词."""
        return self._sys_prompt

    @sysprompt.setter
    def sysprompt(self, newprompt: str):
        """修改本agent的系统提示词."""
        self._sys_prompt = newprompt

    @property
    def description(self) -> str:
        """获取本agent的描述."""
        return self._description

    @description.setter
    def description(self, newdescription: str):
        """修改本agent的描述."""
        self._description = newdescription

    #--------------------------------------------
    #  将prompt消息转换为消息对象
    #--------------------------------------------
    def _get_sysprompt_msg(self) -> Msg:
        return Msg("system", content=self._sys_prompt, role="system")

    #--------------------------------------------
    #  获取某个发送端 agent 最大自动回复数量
    #--------------------------------------------
    def get_max_num_auto_reply(self, sender_name:Optional[str] = None) -> int:
        """获取对应的那个用户的 最大自动回复数 max_num_auto_reply，没有sender就去默认值 ."""
        return self._max_num_auto_reply if sender_name is None else self._max_num_auto_reply_dict[sender_name]
    
    #--------------------------------------------
    #  更新连续自动回复的最大次数。 
    #  没携带 sender_name  就是更新所有的，带了 sender_name 就是更新指定的  
    #  value是本次要更新的最大次数
    #--------------------------------------------
    def update_max_num_auto_reply(self, value: int, sender_name: Optional[str] = None):
        """Update the maximum number of consecutive auto replies.

        Args:
            value (int): the maximum number of consecutive auto replies.
            sender (Agent): when the sender is provided, only update the max_consecutive_auto_reply for that sender.
        """
        if sender_name is None:
            self._max_num_auto_reply = value # 更新最大自动回复数目
            for k in self._max_num_auto_reply_dict: # 变量 dict，修改所有回复对象(请求发送者，需要回响应)的最大数目
                self._max_num_auto_reply_dict[k] = value
        else:
            # 如果sender不为空，那么就是更新指定的发送者的最大自动回复数目
            self._max_num_auto_reply_dict[sender_name] = value
 
    #--------------------------------------------
    #  重置 auto reply counter 自动回复计数
    #--------------------------------------------
    def reset_num_auto_reply_counter(self, sender_name: Optional[str] = None):
        """Reset the consecutive_auto_reply_counter of the sender."""
        if sender_name is None:
            self._auto_reply_counter.clear()
        else:# 把对某个人自动回复清空
            self._auto_reply_counter[sender_name] = 0

    #--------------------------------------------
    #  检测llm的配置
    #--------------------------------------------
    def _validate_llm_config_and_load(self, llm_config, tool_manager):
        assert llm_config in (None, False) or isinstance(
            llm_config, dict
        ), "llm_config must be a dict or False or None."
        # llm_config 为空就表示没有 大模型配置
        if llm_config is None:
            llm_config = self.DEFAULT_CONFIG
        self.llm_config = self.DEFAULT_CONFIG if llm_config is None else llm_config
        # TODO: more complete validity check
        # 有内容就应该是这种格式
        if self.llm_config in [{}, {"config_list": []}, {"config_list": [{"model": ""}]}]:
            raise ValueError(
                "need a non-empty 'model' either in 'llm_config' or in each config of 'config_list'."
            )
        LOG_INFO("_validate_llm_config_and_load ------------>\n llm_config: ", self.llm_config)

        #-===================================================================================
        #根据配置,如果没有配置 那么就是none表示不使用大模型处理，例如userAgent，否则就使用模型包装器
        #-===================================================================================
        self.model = None if self.llm_config is False else ModelChatWrapper(**self.llm_config, tool_manager=tool_manager)

    #--------------------------------------------
    #  注册 agent的生成响应的处理函数
    #--------------------------------------------
    def register_reply(
        self,
        trigger: Union[Type[Operator], str, Operator, Callable[[Operator], bool], List],
        reply_func: Callable,
        position: int = 0,
        config: Optional[Any] = None,
        reset_config: Optional[Callable] = None,
        *,
        ignore_sync_in_async_chat: bool = False,
        ignore_async_in_sync_chat: bool = False,
        remove_other_reply_funcs: bool = False,
    ):
        LOG_INFO("BaseAgent-----------register_reply: reply_func",reply_func)
        #  config 可以指定大模型配置，这样在reply 中可以使用指定的大模型进行想要的处理。
        if not isinstance(trigger, (type, str, Operator, Callable, list)):
            raise ValueError("trigger must be a class, a string, an Operator, a callable or a list.")
        if remove_other_reply_funcs:
            self._reply_func_list.clear()
        # 根据postion插入具体的位置
        # 如果有 ignore_sync_in_async_chat 异步里面忽略同步函数标识，并且当前函数是同步函数，则忽略 
        # 如果有 ignore_async_in_sync_chat 同步里面忽略异步函数标识，并且当前函数是异步函数，则忽略
        self._reply_func_list.insert(
            position,
            {
                "trigger": trigger,
                "reply_func": reply_func,
                "config": copy.copy(config),
                "init_config": config,
                "reset_config": reset_config,
                "ignore_sync_in_async_chat": ignore_sync_in_async_chat and not inspect.iscoroutinefunction(reply_func),
                "ignore_async_in_sync_chat": ignore_async_in_sync_chat and inspect.iscoroutinefunction(reply_func),
            },
        )

    #--------------------------------------------
    #  替换处理 reply 函数
    #--------------------------------------------
    def replace_reply_func(self, old_reply_func: Callable, new_reply_func: Callable):
        """Replace a registered reply function with a new one.

        Args:
            old_reply_func (Callable): the old reply function to be replaced.
            new_reply_func (Callable): the new reply function to replace the old one.
        """
        for f in self._reply_func_list:
            if f["reply_func"] == old_reply_func:
                f["reply_func"] = new_reply_func

    #--------------------------------------------
    #  注册 hook 的函数，
    #  这个注册函数是将构造挂载到 hook_lists上
    #--------------------------------------------
    def register_hook(
            self, 
            hookable_method: str, 
            hook: Callable, 
            position: int = 0,
            remove_other_hook_funcs: bool = False
    ):
        LOG_INFO(f"BaseAgent-----------register_hook: {hookable_method} hook",hook)
        # 如果挂载点找不到返回失败
        assert hookable_method in self.hook_lists, f"{hookable_method} is not a hookable method."
        hook_list = self.hook_lists[hookable_method] # 获取挂载点
        assert hook not in hook_list, f"{hook} is already registered as a hook." #如果已经有了这个函数返回失败
        # remove_other_hook_funcs 清除其他的hook 函数
        if remove_other_hook_funcs:
            hook_list.clear() # 将当前指定挂载点的函数清空
        if position != 0:
            hook_list.insert(position,hook) # 挂载钩子函数到指定位置
        else:
            hook_list.append(hook) # 挂载钩子函数到末尾

    #--------------------------------------------
    #  将当前消息，保存历史记录，用于记忆，这个消息可以是列表
    #--------------------------------------------
    def _save_history(self, message: Union[Sequence[Msg], Msg, None]):
        # 将消息加到最后
        if self.use_memory:
            self.memory.add(message)
            LOG_INFO("BaseAgent::================_save_history==memory.add:\n",message)

    #--------------------------------------------
    #  将当前消息，保存历史记录，用于记忆，这个消息可以是列表，默认取全部
    #  获取记忆，到llm处理过程消息过程中会。 执行保存记忆的操作，所以这里不能获取指针出去，应为add记忆的时候会变
    #--------------------------------------------
    def _get_history(self, 
                     recent_n: Optional[int] = None, 
                     filter_func: Optional[Callable[[int, dict], bool]] = None
                     )->list:
        # 将消息加到最后
        if self.use_memory:
            historylist = self.memory.get_memory(recent_n=recent_n, filter_func=filter_func)
            newhistorylist = copy.deepcopy(historylist)
            LOG_INFO("BaseAgent::================_get_history:\n",newhistorylist)
            return newhistorylist

    #--------------------------------------------
    #  打印接收到的消息
    #--------------------------------------------
    def _print_rcv_message(self, messages: Union[str, List[Dict[str, Any]]])-> Union[str, List[Dict[str, Any]]]:
        LOG_INFO("BaseAgent::==================_print_rcv_message")
        # print the message received
        if isinstance(messages,str):
            LOG_INFO("recv a message:\n ",messages, flush=True)
            return messages

        if isinstance(messages,list):
            LOG_INFO("recv message from ",messages[-1].name, " (to ", f"{self.name}):\n", flush=True)
        else:
            #LOG_INFO("recv message from ",messages, flush=True)
            LOG_INFO("recv message from ",messages["name"], " (to ", f"{self.name}):\n", flush=True)
        LOG_INFO("message:\n ",messages, flush=True)
        return messages
   
    #--------------------------------------------
    #  对接收到的消息进行预处理,全部转成list[Dict[str, Any]] 格式
    #--------------------------------------------
    def _preprocess_rcv_message(self, messages: Union[str, List[Dict[str, Any]]]) -> Union[List[Dict[str, Any]]]:
        LOG_INFO("BaseAgent::==================_preprocess_rcv_message")
        #----------------------------------------------------------------------
        # 如果传入的是str那么说明这个消息是用户发出去的，其他agent处理的都是Msg消息
        #----------------------------------------------------------------------
        if isinstance(messages,str):
            x = Msg(name="user",role="user", content=messages)
            # print the message received
            LOG_INFO("recv message from ",x.name, " (to ", f"{self.name}):\n", flush=True)
            LOG_INFO("message:\n ",x, flush=True)
            x = [x]

        if isinstance(messages,List):
            x =  messages
        else:
            x = [messages]

        return x

    #--------------------------------------------
    #  在发送消息之前将 ModelResponse 处理成 Msg 或者 流式 Generator
    #--------------------------------------------
    def _process_send_message(self, message: Union[ModelResponse, List[Dict], None])-> Union[List[Dict], Generator, None]:
        LOG_INFO("BaseAgent::==================_process_send_message")
        '''
        #----------------------------------------------------------------------
        #  下面下面都是llm返回的 ModelResponse 格式消息   
        #   里面有 1：stream 消息。 2：普通的消息 3：function call的消息
        #----------------------------------------------------------------------
        elif isinstance(message, ModelResponse):
            # 如果是流式的，返回流式生成器
            if message.stream is not None:
                message = message.stream # 返回流式生成器
                LOG_INFO("baseAgent::==================_process_send_message-----messages",message,flush=True)
            
            # 如果是function call的响应
            elif message.is_funcall is True:
                # 获取function call的响应 , 将role设置为 tool_calls 标识这是一个 function call的消息
                message = Msg(name = self.name, role="tool", content = "", message = message.text) # 返回 Msg 消息
            #其他的都是llm返回的给用户消息
            else:
                #返回 Msg 消息
                message = Msg(name = self.name, role="assistant", content = message.text) # 返回 Msg 消息
        '''
        #----------------------------------------------------------------------
        # 消息为空的是要退出的 exit
        #----------------------------------------------------------------------
        if message is None:
            return None # 返回 None
        #----------------------------------------------------------------------
        #  下面消息是 human replay的消息，直接就是msg消息，不用处理,
        #   或者多个msg消息，也是直接透传（这种情况是 userAgent，没有大模型处理的时候，对收到的消息用户不回复，自动情况下透传）
        #----------------------------------------------------------------------
        else:
            #返回 Msg 消息
            message = message # 返回 Msg 消息

        return message

    #--------------------------------------------
    #  process_received_message:  
    #   
    #  Hook point 接收到消息的时候 消息处理的hook点
    #
    #--------------------------------------------
    def _process_received_message(self, messages: Union[str, List[Dict[str, Any]]]):
        LOG_INFO("BaseAgent::==================_process_received_message")
        """接收到了消息后对消息进行处理。"""
        ''' 可以添加对 接收到的消息的处理 
        '''
        hook_list = self.hook_lists["process_received_message"]
        # 没有钩子直接返回
        if len(hook_list) == 0:
            return messages  # No hooks registered.
        
        # 遍历 hook函数，处理 发送之前的消息
        for hook in hook_list:
            # 运行钩子函数
            messages = hook(messages=messages)
        return messages

    #--------------------------------------------
    #  process_all_messages_before_reply:  
    #   
    #  Hook point  生成响应之前的消息处理点 消息处理的hook点
    #
    #--------------------------------------------
    def _process_all_messages_before_reply(self, messages: List[Dict[str, Any]]):
        LOG_INFO("BaseAgent::==================process_all_messages_before_reply")
        """在将消息发送之前对其进行处理。"""
        hook_list = self.hook_lists["process_all_messages_before_reply"]
        # 没有钩子直接返回
        if len(hook_list) == 0:
            return messages  # No hooks registered.
        
        # 遍历 hook函数，处理 发送之前的消息
        for hook in hook_list:
            # 运行钩子函数
            messages = hook(messages=messages)
        return messages

    #--------------------------------------------
    #  process_message_before_send:  
    #   
    #  Hook point  发送消息之前 的消息处理点, ModelResponse 可能是一个stream的Generator 消息处理的hook点
    #
    #--------------------------------------------
    def _process_message_before_send(self, message: Union[ModelResponse, List[Dict], None]) -> Union[None, List[Dict], Generator]:
        LOG_INFO("BaseAgent::==================_process_message_before_send")
        """在将消息发送之前对其进行处理。"""
        hook_list = self.hook_lists["process_message_before_send"]
        # 没有钩子直接返回
        if len(hook_list) == 0:
            return message  # No hooks registered.
        
        # 遍历 hook函数，处理 发送之前的消息
        for hook in hook_list:
            # 运行钩子函数
            message = hook(message=message)

        #x = Msg(self.name, messages.text, role="assistant")
        return message

    #--------------------------------------------
    #  调用回复函数，并在需要时将生成的回复广播给所有受众。 
    #  __call__ 这个函数默认调用的是异步的 消息式应答，非流式应答
    #  agent使用这个接口接收发给自己的消息  
    #  入参是任意(可为0)数量的位置参数（按位置来） 和任意数量(可为0)的关键字参数  
    #--------------------------------------------
    def __call__(self, *args: Any, **kwargs: Any) -> Union[Generator[Tuple[bool, str], None, None], Dict[str, Any], None]:
        LOG_INFO("BaseAgent::==================__call__")
        # 默认使用 asyncio.run() 来调用异步函数
        res = asyncio.run(self.a_generate_response(*args, **kwargs))

        # 如果有听众，则广播给听众，这个消息是不用回响应的
        #if self._audience is not None:
        #    self._broadcast_to_audience(res)
        LOG_INFO("BaseAgent::==================__call__:res:",res)
        return res

    #========================================
    #  处理 toolcall 的逻辑,z这部分逻辑是为了处理 llm放到 function call响应，
    #  收到的响应如果是function call的话，那么就继续调用tools的reply，获取tool调用结果，
    #  然后继续调用llm获取最终响应,直到最终没有tools调用为止
    #========================================
    def _handle_toolcall(self, reply_func_tuple, messages, **kwargs):
        """
        处理 toolcall 的逻辑，抽离自 generate_response 函数中的工具调用逻辑。
        """
        LOG_INFO("BaseAgent::==================_handle_toolcall:begin:")
        final = False  # 是否是最终消息, 跳出for 循环
        retry_count = 0  # 初始化重试次数

        while retry_count < self._max_num_tools_call:
            # 1: 运行 generate_llm_reply, 返回值第一个是否有reponse，第二个参数是repsonse消息
            final, funcall_reply = self.generate_llm_reply(messages=messages, config=reply_func_tuple["config"],**kwargs)
            if not final:  # 如果reply 消息为空,那么就直接返回
                return False, None
        
            # 如果有reply消息，并且不是function call的消息
            if funcall_reply.role != "tool":
                return True, funcall_reply
            
            # 如果是function call的消息，那么就继续调用tools的reply，获取tool调用结果
            # 如果是toolcall，那么就继续while循环
            # 将 llm的响应 reply 添加到 用户消息messages中去，作为tools replay的入口消息
            #funcall_reply_msg = Msg(name = self.name, role="tool", content = "", message = funcall_reply.text)
            #messages.append(reply)  # 这个是function call的消息,作为tools replay的入口消息

            # 2: 运行 generate_tool_calls_reply, 传入的消息是 function call的消息，用户消息没传入，tools call不需要
            final, reply = self.generate_tool_calls_reply(messages=[funcall_reply], config=reply_func_tuple["config"], **kwargs)

            if not final:  # 如果tool call没有返回tool call响应而是一个 none那么这个函数调用循环流程就会结束，说明函数调用出错了
                return True, reply
            else:
                # 如果tool call 返回了 ture，说明生成了 tools call调用的响应，则需要大模型进一步处理
                messages = [reply]

            retry_count += 1  # 重试次数加1 , 根据限制的次数处理

        return False, None  # 表示没有生成最终消息

    #========================================
    #  处理 toolcall 的逻辑,z这部分逻辑是为了处理 llm放到 function call响应，
    #  收到的响应如果是function call的话，那么就继续调用tools的reply，获取tool调用结果，
    #  然后继续调用llm获取最终响应,直到最终没有tools调用为止
    #========================================
    async def _a_handle_toolcall(self, reply_func_tuple, messages, **kwargs):
        """
        处理 toolcall 的逻辑，抽离自 generate_response 函数中的工具调用逻辑。
        """
        final = False  # 是否是最终消息, 跳出for 循环
        retry_count = 0  # 初始化重试次数

        while retry_count < self._max_num_tools_call:
            # 1: 运行 generate_llm_reply, 返回值第一个是否有reponse，第二个参数是repsonse消息
            final, funcall_reply = await self.a_generate_llm_reply(messages=messages, config=reply_func_tuple["config"], **kwargs)
            if not final:  # 如果reply 消息为空,那么就直接返回
                return False, None
        
            # 如果有reply消息，并且不是function call的消息
            if funcall_reply.role != "tool":
                return True, funcall_reply
            
            # 如果是function call的消息，那么就继续调用tools的reply，获取tool调用结果
            # 如果是toolcall，那么就继续while循环
            # 将 llm的响应 reply 添加到 用户消息messages中去，作为tools replay的入口消息
            #funcall_reply_msg = Msg(name = self.name, role="tool", content = "", message = funcall_reply.text)
            #messages.append(reply)  # 这个是function call的消息,作为tools replay的入口消息

            # 2: 运行 generate_tool_calls_reply, 传入的消息是 function call的消息，用户消息没传入，tools call不需要
            final, reply = await self.a_generate_tool_calls_reply(messages=[funcall_reply], config=reply_func_tuple["config"], **kwargs)

            if not final:  # 如果是最终消息 ， tool calls中目前应该没有下面这种情况出现
                return True, reply
            else:
                # 如果不是最终消息，并且有reply消息，这个消息是function call的消息的结果，将他添加到用户消息中
                if reply:
                    messages = [reply]

            retry_count += 1  # 重试次数加1 , 根据限制的次数处理

        return False, None  # 表示没有生成最终消息
    
    #========================================
    #同步 消息式 generate_response   
    # 返回值 第一个 Generator[Tuple[bool, str], None, None]  
    #               生成器每次生成的值是一个元组，元组的第一个元素是一个布尔值 (bool)，第二个元素是一个字符串 (str) 
    #               None（第二个类型）：指定生成器的 send 方法的参数类型（这里是 None，表示不接受任何值通过 send 方法传递给生成器） 
    #               None（第三个类型）：指定生成器的 return 语句的返回类型（这里是 None，表示生成器正常结束时不返回任何值）
    #========================================
    def generate_response(
            self,
            messages: Union[str, List[Dict[str, Any]],None],
            **kwargs: Any,
        ) -> Union[Generator[Tuple[bool, str], None, None], Dict[str, Any], None]:
        LOG_INFO("BaseAgent::==================generate_response")

        #==============================================
        # 进入此函数就不是流式输出，所以 stream参数必须是False
        #==============================================
        kwargs['stream'] = False

        # 消息不能为空
        if messages is None:
            raise ValueError(
                "message can not None. in a_generate_response!"
            )
        
        #--------------------------------------------------
        # 对收到的消息进行处理，这个是接收到消息的时候的挂载点 
        # -------------------------------------------------
        messages = self._process_received_message(messages)

        #--------------------------------------------------
        # 对消息进行处理
        # -------------------------------------------------
        # 如果消息为 none，则获取默认的消息
        #if messages is None:
            #如果传入的消息为空则获取默认消息 ,这个后面增加
        #    pass

        #--------------------------------------------------
        # 在生成响应之前进行处理，这个是生成响应之前时候的挂载点 
        # -------------------------------------------------
        messages = self._process_all_messages_before_reply(messages)

        #--------------------------------------------------
        #逐个处理使用register_reply 注册到， 生成响应的的钩子函数 _reply_func_list
        #--------------------------------------------------
        #LOG_INFO("BaseAgent::========aaaaabbb==========generate_response:_reply_func_list",self._reply_func_list)
        message = None # 初始化消息为空
        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            #LOG_INFO("BaseAgent::========aaaaaa==========generate_response:reply_func",reply_func)
            #如果参数中有 exclude的键值，则跳过这些排除的钩子函数
            if "exclude" in kwargs and reply_func in kwargs["exclude"]:
                continue
            if reply_func_tuple["ignore_async_in_sync_chat"] is True: # 如果有忽略异步函数标志则contiue
                continue
            if inspect.iscoroutinefunction(reply_func): # 同步函数不能调用异步函数，所以异步函数跳过
                continue

            if self._match_trigger(reply_func_tuple["trigger"], None):# 根据触发点触发，默认None触发
                #如果当前函数是 tools调用函数 a_generate_llm_reply, 则开始while循环
                #LOG_INFO("BaseAgent::========aaaaa221==========generate_response:generate_llm_reply.__func__ :", self.generate_llm_reply.__func__ )
                #LOG_INFO("BaseAgent::========aaaaa222==========generate_response:ture or false:",reply_func == self.generate_llm_reply.__func__)
                if self._hide_toolscall and reply_func == self.generate_llm_reply.__func__:
                    
                    # 处理 tool call 逻辑
                    final, message = self._handle_toolcall(reply_func_tuple=reply_func_tuple, messages = messages, **kwargs)
                    
                    # 如果是最终消息，跳出for循环 遍历
                    if final:
                        #LOG_INFO("BaseAgent::========aaaaa2233==========generate_response:final message:",message)
                        break

                else:
                    #LOG_INFO("BaseAgent::========aaaaa333==========generate_response:reply_func:",reply_func)
                    # 符合触发要求， 调用触发函数
                    final, reply = reply_func(self, messages=messages, config=reply_func_tuple["config"],**kwargs)                
                    
                    if final: # 如果是最终消息
                        message = reply
                        # 如果是最终消息，判断是否是tool消息，如果是tool消息，那么就继续循环
                        if isinstance(message,Msg) and message.role == "tool":
                            continue
                        else:
                            break # 跳出 遍历不继续循环了
                
        #--------------------------------------------------
        # 如果 上面的挂载点没有生成消息，那么就返回未处理的默认消息
        # -------------------------------------------------
        if message is None:
            message = messages

        #--------------------------------------------------
        # 对发出去的消息进行预处理，这个是发送消息之前的的挂载点
        # -------------------------------------------------
        message = self._process_message_before_send(message)

        return message

    #========================================
    # 异步 消息式 generate_response 
    # 
    #========================================
    # 返回结果可能是 msg，或者 none
    async def a_generate_response(
        self,
        messages: Union[str, List[Dict[str, Any]],None],
        **kwargs: Any,
    ) -> Union[Generator[Tuple[bool, str], None, None], Dict[str, Any], None]:
        LOG_INFO("BaseAgent::==================a_generate_response")

        #==============================================
        # 进入此函数就不是流式输出，所以 stream参数必须是False
        #==============================================
        kwargs['stream'] = False

        # 消息不能为空
        if messages is None:
            raise ValueError(
                "message can not None. in a_generate_response!"
            )
        #--------------------------------------------------
        # 对收到的消息进行处理，这个是接收到消息的时候的挂载点 
        # -------------------------------------------------
        messages = self._process_received_message(messages)

        #--------------------------------------------------
        # 对消息进行处理
        # -------------------------------------------------
        # 如果消息为 none，则获取默认的消息
        #if messages is None:
            #如果传入的消息为空则获取默认消息
        #    pass

        #--------------------------------------------------
        # 在生成响应之前进行处理，这个是生成响应之前时候的挂载点 
        # -------------------------------------------------
        messages = self._process_all_messages_before_reply(messages)

        #--------------------------------------------------
        #逐个处理使用register_reply 注册到， 生成响应的的钩子函数 _reply_func_list
        #--------------------------------------------------
        message = None # 初始化消息为空
        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if "exclude" in kwargs and reply_func in kwargs["exclude"]:
                continue
            if reply_func_tuple["ignore_sync_in_async_chat"] is True: # 如果有忽略同步函数标志则contiue
                continue

            if self._match_trigger(reply_func_tuple["trigger"], None):
                #如果当前函数是 tools调用函数 a_generate_llm_reply, 则开始while循环
                if self._hide_toolscall and reply_func  == self.a_generate_llm_reply.__func__:

                    # 处理 tool call 逻辑
                    final, message = await self._a_handle_toolcall(reply_func_tuple, messages, **kwargs)                    
                    
                    # 如果是最终消息，跳出for循环
                    if final:
                        break

                else:
                    # 如果不是 generate_tool_calls_reply就继续for循环
                    if inspect.iscoroutinefunction(reply_func):
                        final, reply = await reply_func(
                            self, messages=messages, config=reply_func_tuple["config"],**kwargs
                        )
                    else:
                        final, reply = reply_func(self, messages=messages, config=reply_func_tuple["config"],**kwargs)

                    if final: # 如果是最终消息
                        message = reply
                        # 如果是最终消息，判断是否是tool消息，如果是tool消息，那么就继续循环
                        if isinstance(message,Msg) and message.role == "tool":
                            continue
                        else:
                            break # 跳出 遍历不继续循环了

        #--------------------------------------------------
        # 如果 上面的挂载点没有生成消息，那么就返回未处理的默认消息
        # -------------------------------------------------
        if message is None:
            message = messages

        #--------------------------------------------------
        #  调用完大模型后，返回的结果是 ModelResponse 
        #--------------------------------------------------
        # 对发出去的消息进行预处理，这个是发送消息之前的的挂载点
        # -------------------------------------------------
        abc = self._process_message_before_send(message)
        LOG_INFO("BaseAgent::==========a_generate_response: message:\n",abc)
        return abc

    #同步 generate_response 
    def stream(
            self,
            messages: Union[str, List[Dict[str, Any]],None],
            **kwargs: Any,
        ) -> Union[Generator[Tuple[bool, str], None, None], Dict[str, Any], None]:
        LOG_INFO("BaseAgent::==================stream")

        #==============================================
        # 进入此函数就是要求流式输出，所以 stream参数必须是True
        #==============================================
        kwargs['stream'] = True

        # 消息不能为空
        if messages is None:
            raise ValueError(
                "message can not None. in stream!"
            )
        
        #--------------------------------------------------
        # 对收到的消息进行处理，这个是接收到消息的时候的挂载点 
        # -------------------------------------------------
        messages = self._process_received_message(messages)

        #--------------------------------------------------
        # 对消息进行处理
        # -------------------------------------------------
        # 如果消息为 none，则获取默认的消息
        #if messages is None:
            #如果传入的消息为空则获取默认消息 ,这个后面增加
        #    pass

        #--------------------------------------------------
        # 在生成响应之前进行处理，这个是生成响应之前时候的挂载点 
        # -------------------------------------------------
        messages = self._process_all_messages_before_reply(messages)

        #--------------------------------------------------
        #逐个处理使用register_reply 注册到， 生成响应的的钩子函数 _reply_func_list
        #--------------------------------------------------
        message = None # 初始化消息为空
        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            #如果参数中有 exclude的键值，则跳过这些排除的钩子函数
            if "exclude" in kwargs and reply_func in kwargs["exclude"]:
                continue
            if reply_func_tuple["ignore_async_in_sync_chat"] is True: # 如果有忽略异步函数标志则contiue
                continue
            if inspect.iscoroutinefunction(reply_func): # 同步函数不能调用异步函数，所以异步函数跳过
                continue
            if self._match_trigger(reply_func_tuple["trigger"], None):# 根据触发点触发，默认None触发
                #如果当前函数是 tools调用函数 a_generate_llm_reply, 则开始while循环
                if self._hide_toolscall and reply_func  == self.generate_llm_reply.__func__:
                    # 处理 tool call 逻辑
                    final, message = self._handle_toolcall(reply_func_tuple, messages, **kwargs)
                    
                    # 如果是最终消息，跳出for循环
                    if final:
                        break

                else:
                    # 符合触发要求， 调用触发函数   no_use 参数是为了统一接口, 只有特点接口才需要这个参数
                    final, reply = reply_func(self, messages=messages, config=reply_func_tuple["config"], **kwargs)                
                    
                    if final: # 如果是最终消息
                        message = reply
                        # 如果是最终消息，判断是否是tool消息，如果是tool消息，那么就继续循环
                        if isinstance(message,Msg) and message.role == "tool":
                            continue
                        else:
                            break # 跳出 遍历不继续循环了
                
        #--------------------------------------------------
        # 如果 上面的挂载点没有生成消息，那么就返回未处理的默认消息
        # -------------------------------------------------
        if message is None:
            message = messages

        #--------------------------------------------------
        # 对发出去的消息进行预处理，这个是发送消息之前的的挂载点
        # -------------------------------------------------
        message = self._process_message_before_send(message)
        LOG_INFO("BaseAgent::==========stream: message",message)
        return message

    #异步 
    # 返回结果可能是 生成器，msg，或者 none
    async def a_stream(
        self,
        messages: Union[str, List[Dict[str, Any]],None],
        **kwargs: Any,
    ) -> Union[Generator[Tuple[bool, str], None, None], Dict[str, Any], None]:
        LOG_INFO("BaseAgent::==================a_stream")

        #==============================================
        # 进入此函数就是要求流式输出，所以 stream参数必须是True
        #==============================================
        kwargs['stream'] = True

        # 消息不能为空
        if messages is None:
            raise ValueError(
                "message can not None. in a_stream!"
            )
        #--------------------------------------------------
        # 对收到的消息进行处理，这个是接收到消息的时候的挂载点 
        # -------------------------------------------------
        messages = self._process_received_message(messages)

        #--------------------------------------------------
        # 对消息进行处理
        # -------------------------------------------------
        # 如果消息为 none，则获取默认的消息
        #if messages is None:
            #如果传入的消息为空则获取默认消息
        #    pass

        #--------------------------------------------------
        # 在生成响应之前进行处理，这个是生成响应之前时候的挂载点 
        # -------------------------------------------------
        messages = self._process_all_messages_before_reply(messages)

        #--------------------------------------------------
        #逐个处理使用register_reply 注册到， 生成响应的的钩子函数 _reply_func_list
        #--------------------------------------------------
        message = None # 初始化消息为空
        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if "exclude" in kwargs and reply_func in kwargs["exclude"]:
                continue
            if reply_func_tuple["ignore_sync_in_async_chat"] is True: # 如果有忽略同步函数标志则contiue
                continue

            if self._match_trigger(reply_func_tuple["trigger"], None):
                #如果当前函数是 tools调用函数 a_generate_llm_reply, 则开始while循环
                if self._hide_toolscall and reply_func  == self.a_generate_llm_reply.__func__:
                
                    # 处理 tool call 逻辑
                    final, message = await self._a_handle_toolcall(reply_func_tuple, messages, **kwargs)    
                    
                    # 如果是最终消息，跳出for循环
                    if final:
                        break

                else:
                    # 如果不是 generate_tool_calls_reply就继续for循环
                    if inspect.iscoroutinefunction(reply_func):
                        final, reply = await reply_func(
                            self, messages=messages, config=reply_func_tuple["config"], **kwargs
                        )
                    else:
                        final, reply = reply_func(self, messages=messages, config=reply_func_tuple["config"], **kwargs)

                    if final: # 如果是最终消息
                        message = reply
                        # 如果是最终消息，判断是否是tool消息，如果是tool消息，那么就继续循环
                        if isinstance(message,Msg) and message.role == "tool":
                            continue
                        else:
                            break # 跳出 遍历不继续循环了

        #--------------------------------------------------
        # 如果 上面的挂载点没有生成消息，那么就返回未处理的默认消息
        # -------------------------------------------------
        if message is None:
            message = messages

        #--------------------------------------------------
        #  调用完大模型后，返回的结果是 ModelResponse 
        #--------------------------------------------------
        # 对发出去的消息进行预处理，这个是发送消息之前的的挂载点
        # -------------------------------------------------
        abc = self._process_message_before_send(message)
        LOG_INFO("BaseAgent::==========a_stream: message",abc)
        return abc

    #--------------------------------------------
    #  触发点匹配函数。返回值是bool类型的
    #--------------------------------------------
    def _match_trigger(self, trigger: Union[None, str, type, Operator, Callable, List], condition: Union[None, str, type, Callable, List]) -> bool:
        """Check if the sender matches the trigger.

        Args:
            - trigger (Union[None, str, type, Agent, Callable, List]): The condition to match against the sender.
            Can be `None`, string, type, `Agent` instance, callable, or a list of these.
            - sender (Agent): The sender object or type to be matched against the trigger.

        Returns:
            - bool: Returns `True` if the sender matches the trigger, otherwise `False`.

        Raises:
            - ValueError: If the trigger type is unsupported.
        """
        if trigger is None: # 如果 trigger 的内容是 None，那么 condition 是none则为true，否则为false
            return condition is None # 如果没有 trigger也没触发条件，则都是true
        elif isinstance(trigger, str): # 如果trigger 是 str，那么和触发条件对比
            if isinstance(condition, str):
                return trigger == condition # 匹配两者的内容
            else:
                return False #其他的都认为匹配失败
        elif isinstance(trigger, type):
            return isinstance(condition, trigger)
        elif isinstance(trigger, Callable): # 如果是 可调用对象 那么一定要是返回值是 bool类型的调用对象
            rst = trigger(condition)
            assert isinstance(rst, bool), f"trigger {trigger} must return a boolean value."
            return rst
        elif isinstance(trigger, list):# 遍历列表内外调用
            return any(self._match_trigger(t, condition) for t in trigger)
        else:
            raise ValueError(f"Unsupported trigger type: {type(trigger)}")

    #--------------------------------------------
    #  人机互动函数 ，在生成响应之前最早处理的第一个register_reply 的注册函数，
    #  这个函数是用户介入的函数
    #--------------------------------------------
    def check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[bool, Union[Msg, None]]:
        LOG_INFO("check_termination_and_human_reply==================\n")
        reply = ""
        no_human_input_msg = "" # 没有人类输入的消息时候的提示

        # 获取最近的一条消息
        message = messages[-1] if messages else {}
        
        # 获取发送者的名字,消息的name字段就是发送者的名字，默认是user
        sender_name =  message.get("name", "user")

        if self.human_input_mode == "ALWAYS":
            # 如果是一直需要人类输入，那么就获取人类输入
            reply = self.get_human_input(
                colored(f"Provide feedback to {sender_name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: ","yellow")
            )
            # 如果没有人类输入，那么就提示没有输入，有人类输入则为空白字符串
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # 如果人工输入为空，且该消息是终止消息，则我们将终止对话 reply = exit，否则 reply = reply
            reply = reply if reply or not self.is_stop_msg(message) else "exit"

        else:
            # 如果 auto_reply_counter 大于等于 _max_num_auto_reply_dict 最大自动回复次数，那么就需要人类输入
            if self._auto_reply_counter[sender_name] >= self._max_num_auto_reply_dict[sender_name]:
                # 如果人类输入模式是 NEVER ，那么如果达到了最大次数就直接返回 exit
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # 这下面的就是 TERMINATE ，self.human_input_mode == "TERMINATE":
                    terminate = self.is_stop_msg(message)
                    # 如果消息是终止消息 则提示终止消息，用户输入不输入都会结束，否则就会自动回复，除非用户输入exit
                    reply = self.get_human_input(
                        colored(f"Please give feedback to {sender_name}. Press enter or type 'exit' to stop the conversation: ","yellow")
                        if terminate
                        else colored(f"Please give feedback to {sender_name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: ","yellow")
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # 如果人工输入为空，且该消息是终止消息，则我们将终止对话
                    reply = reply if reply or not terminate else "exit"

            # 如果消息是 stop 消息
            elif self.is_stop_msg(message):
                # 如果人类输入模式是 NEVER ，stop消息就直接返回 exit
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    reply = self.get_human_input(
                        colored(f"Please give feedback to {sender_name}. Press enter or type 'exit' to stop the conversation: ","yellow")
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply or "exit"

        # 如果没有 用户的 输入消息，那么就返回 false，和空
        if no_human_input_msg:
            self.output_to_human(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"))
            #iostream.print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # 如果用户输入了 exit 则返回 true 和空，表示停止对话
        if reply == "exit":
            # 重置自动回复计数 _auto_reply_counter
            self._auto_reply_counter[sender_name] = 0
            return True, None  #第二个参数 fasle 目前没啥用处
        
        # 如果用户有输入，或者 针对这个 人的回复的最大次数为0 
        if reply or self._max_num_auto_reply_dict[sender_name] == 0:
            # 用户提供自定义响应，返回 工具结果，表明用户中断
            # reset the consecutive_auto_reply_counter
            self._auto_reply_counter[sender_name] = 0 # 重置自动回复计数
            tool_returns = []
            if message.get("tool_calls", False):
                tool_returns.extend(
                    [
                        {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": "USER INTERRUPTED"}
                        for tool_call in message["tool_calls"]
                    ]
                )

            #response = {"role": "user", "content": reply}
            if tool_returns:
                response = Msg(name="human", role ="user", content=reply, tool_responses = tool_returns)
            else:
                response = Msg(name="human", role ="user", content=reply)
            # 这里返回的response 是一个字典，role 是 user，content 是用户输入的内容,还有可能有 tool_responses

            return True, response

        # _auto_reply_counter 计数  + 1
        self._auto_reply_counter[sender_name] += 1
        if self.human_input_mode != "NEVER":# 如果不是永远不需要人类输入，则输出提示
            self.output_to_human(colored("\n>>>>>>>> USING AUTO REPLY...", "red"))

        return False, None

    #--------------------------------------------
    #  人机互动函数 ，在生成响应之前最早处理的第一个register_reply 的注册函数，
    #  这个函数是用户介入的函数， 异步函数
    #--------------------------------------------
    async def a_check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[bool, Union[Msg, None]]:
        LOG_INFO("a_check_termination_and_human_reply==================\n")
        reply = ""
        no_human_input_msg = "" # 没有人类输入的消息时候的提示

        # 获取最近的一条消息
        message = messages[-1] if messages else {}
        
        # 获取发送者的名字,消息的name字段就是发送者的名字，默认是user
        sender_name =  message.get("name", "user")

        if self.human_input_mode == "ALWAYS":
            # 如果是一直需要人类输入，那么就获取人类输入
            reply = await self.a_get_human_input(
                colored(f"Provide feedback to {sender_name}. Press enter to skip and use auto-reply, or type 'exit' to end the conversation: ","yellow")
            )
            # 如果没有人类输入，那么就提示没有输入，有人类输入则为空白字符串
            no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
            # 如果人工输入为空，且该消息是终止消息，则我们将终止对话 reply = exit，否则 reply = reply
            reply = reply if reply or not self.is_stop_msg(message) else "exit"

        else:
            # 如果 auto_reply_counter 大于等于 _max_num_auto_reply_dict 最大自动回复次数，那么就需要人类输入
            if self._auto_reply_counter[sender_name] >= self._max_num_auto_reply_dict[sender_name]:
                # 如果人类输入模式是 NEVER ，那么如果达到了最大次数就直接返回 exit
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # 这下面的就是 TERMINATE ，self.human_input_mode == "TERMINATE":
                    terminate = self.is_stop_msg(message)
                    # 如果消息是终止消息 则提示终止消息，用户输入不输入都会结束，否则就会自动回复，除非用户输入exit
                    reply = await self.a_get_human_input(
                        colored(f"Please give feedback to {sender_name}. Press enter or type 'exit' to stop the conversation: ","yellow")
                        if terminate
                        else colored(f"Please give feedback to {sender_name}. Press enter to skip and use auto-reply, or type 'exit' to stop the conversation: ","yellow")
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # 如果人工输入为空，且该消息是终止消息，则我们将终止对话
                    reply = reply if reply or not terminate else "exit"

            # 如果消息是 stop 消息
            elif self.is_stop_msg(message):
                # 如果人类输入模式是 NEVER ，stop消息就直接返回 exit
                if self.human_input_mode == "NEVER":
                    reply = "exit"
                else:
                    # self.human_input_mode == "TERMINATE":
                    reply = await self.a_get_human_input(
                        colored(f"Please give feedback to {sender_name}. Press enter or type 'exit' to stop the conversation: ","yellow")
                    )
                    no_human_input_msg = "NO HUMAN INPUT RECEIVED." if not reply else ""
                    # if the human input is empty, and the message is a termination message, then we will terminate the conversation
                    reply = reply or "exit"

        # 如果没有 用户的 输入消息，那么就返回 false，和空
        if no_human_input_msg:
            await self.a_output_to_human(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"))
            #iostream.print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # 如果用户输入了 exit 则返回 true 和空，表示停止对话
        if reply == "exit":
            # 重置自动回复计数 _auto_reply_counter
            self._auto_reply_counter[sender_name] = 0
            return True, None # 用户自动退出的时候 msg为空 
               
        # 如果用户有输入，或者 针对这个 人的回复的最大次数为0 
        if reply or self._max_num_auto_reply_dict[sender_name] == 0:
            # 用户提供自定义响应，返回 工具结果，表明用户中断
            # reset the consecutive_auto_reply_counter
            self._auto_reply_counter[sender_name] = 0 # 重置自动回复计数
            tool_returns = []
            if message.get("tool_calls", False):
                tool_returns.extend(
                    [
                        {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": "USER INTERRUPTED"}
                        for tool_call in message["tool_calls"]
                    ]
                )

            #response = {"role": "user", "content": reply}
            if tool_returns:
                response = Msg(name="human", role ="user", content=reply, tool_responses = tool_returns)
                #response["tool_responses"] = tool_returns
            else:
                response = Msg(name="human", role ="user", content=reply)
            
            # 这里返回的response 是一个字典，role 是 user，content 是用户输入的内容,还有可能有 tool_responses
            return True, response

        # _auto_reply_counter 计数  + 1
        self._auto_reply_counter[sender_name] += 1
        if self.human_input_mode != "NEVER":# 如果不是永远不需要人类输入，则输出提示
            await self.a_output_to_human(colored("\n>>>>>>>> USING AUTO REPLY...", "red"))

        return False, None

    #获取tool_response 中的content内容
    def _str_for_tool_response(self, tool_response):
        return str(tool_response.get("content", ""))
    
    # 将json字符串格式化为字典
    @staticmethod
    def _format_json_str(jstr):
        """Remove newlines outside of quotes, and handle JSON escape sequences.

        1. this function removes the newline in the query outside of quotes otherwise json.loads(s) will fail.
            Ex 1:
            "{\n"tool": "python",\n"query": "print('hello')\nprint('world')"\n}" -> "{"tool": "python","query": "print('hello')\nprint('world')"}"
            Ex 2:
            "{\n  \"location\": \"Boston, MA\"\n}" -> "{"location": "Boston, MA"}"

        2. this function also handles JSON escape sequences inside quotes,
            Ex 1:
            '{"args": "a\na\na\ta"}' -> '{"args": "a\\na\\na\\ta"}'
        """
        result = []
        inside_quotes = False
        last_char = " "
        for char in jstr:
            if last_char != "\\" and char == '"':
                inside_quotes = not inside_quotes
            last_char = char
            if not inside_quotes and char == "\n":
                continue
            if inside_quotes and char == "\n":
                char = "\\n"
            if inside_quotes and char == "\t":
                char = "\\t"
            result.append(char)
        return "".join(result)
    
    async def a_execute_function(self, func_call):
        """Execute an async function call and return the result.

        Override this function to modify the way async functions and tools are executed.

        Args:
            func_call: a dictionary extracted from openai message at key "function_call" or "tool_calls" with keys "name" and "arguments".

        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".
        """
        LOG_INFO("a_execute_function==================\n")
        # 获取函数的名称，这个对应的是agent tool中的实例
        func_name = func_call.get("name", "")
        LOG_INFO("a_execute_function==================func_name:",func_name)
        #判断当前是否有此工具
        is_exist_func = self.is_agent_tool_exist(func_name)
        LOG_INFO("  ==================is_exist_func:",is_exist_func)

        is_exec_success = False
        # 如果有此工具
        if is_exist_func:
            # 从类似 JSON 的字符串中提取参数并将其放入字典中。
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            LOG_INFO("a_execute_function==================input_string:",input_string)
            try:
                arguments = json.loads(input_string)
                LOG_INFO("a_execute_function==================arguments:",arguments)
            except json.JSONDecodeError as e:
                arguments = None
                content = f"Error: {e}\n You argument should follow json format."

            # Try to execute the function
            if arguments is not None:
                LOG_INFO(colored(f"\n>>>>>>>> EXECUTING ASYNC FUNCTION {func_name}...", "magenta"),flush=True)
                try:
                    # 如果参数不为空，那么就调用函数
                    content = self.call_tool( func_name, arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    #--------------------------------------------
    #异步执行函数调用
    #--------------------------------------------
    async def _a_execute_tool_call(self, tool_call):

        #id = tool_call["id"]
        function_call = tool_call.get("function", {})
        LOG_INFO("a_execute_tool_call==================function_call:",function_call)

        # 执行函数调用
        _, func_return = await self.a_execute_function(function_call)

        # 从函数返回结果中获取 content
        content = func_return.get("content", "")
        if content is None:
            content = ""

        # 从函数返回结果中获取 function name
        func_name = func_return.get("name", None)
        if func_name is None:
            func_name = ""
        
        # 如果有 tool_id字段，那么就返回 tool_call_id,否则就把function的名字返回    
        tool_call_id = tool_call.get("id", None)
        if tool_call_id is not None:
            tool_call_response = {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": content,
            }
        else:
            # openai有 tool_id字段，而qwen没有这个字段，但是qwen有 name字段，这个是function的名字
            # This is to make the tool call object compatible with Mistral API.
            tool_call_response = {
                "name": func_name,
                "role": "tool",
                "content": content,
            }

        return tool_call_response
    
    def execute_function(self, func_call, verbose: bool = False) -> Tuple[bool, Dict[str, str]]:
        """Execute a function call and return the result.

        Override this function to modify the way to execute function and tool calls.

        Args:
            func_call: a dictionary extracted from openai message at "function_call" or "tool_calls" with keys "name" and "arguments".

        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".

        "function_call" deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        """
        func_name = func_call.get("name", "")
        #判断当前是否有此工具
        is_exist_func = self.is_agent_tool_exist(func_name)

        is_exec_success = False
        if is_exist_func:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                content = f"Error: {e}\n You argument should follow json format."

            # Try to execute the function
            if arguments is not None:
                LOG_INFO(colored(f"\n>>>>>>>> EXECUTING ASYNC FUNCTION {func_name}...", "magenta"),flush=True)
                try:
                    content = self.call_tool( func_name, arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            content = f"Error: Function {func_name} not found."

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    def generate_tool_calls_reply(
        self,
        messages: Optional[List[Dict]] = None,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[bool, Union[Dict, None]]:
        """Generate a reply using tool call."""
        LOG_INFO("generate_tool_calls_reply==================\n")

        if messages is None:
            LOG_INFO("generate_tool_calls_reply=================messages none!\n")
            return False, None
        
        #获取最近一条消息
        message = messages[-1]
        tool_returns = []

        # 如果消息中的 message 键的键值为 空，那么就返回 false 和空
        if not message.get("message"):
            LOG_INFO("generate_tool_calls_reply=============message is none or message's value is none!\n")
            return False, None

        tool_message = message.get("message", [])
        LOG_INFO("generate_tool_calls_reply=================tool_message:",tool_message)
        for tool_call in tool_message.get("tool_calls", []):
            # 获取 function 
            function_call = tool_call.get("function", {})
            func_name = function_call.get("name", None)
            if func_name is None:
                continue

            is_exist_func = self.is_agent_tool_exist(func_name)
            # 如果有此工具
            if is_exist_func:

                _, func_return = self.execute_function(function_call)
                    
                content = func_return.get("content", "")
                if content is None:
                    content = ""
                tool_call_id = tool_call.get("id", None)
                if tool_call_id is not None:
                    tool_call_response = {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "content": content,
                    }
                else:
                    # openai有 tool_id字段，而qwen没有这个字段，但是qwen有 name字段，这个是function的名字
                    # This is to make the tool call object compatible with Mistral API.
                    tool_call_response = {
                        "name": func_name,
                        "role": "tool",
                        "content": content,
                    }
                tool_returns.append(tool_call_response)

        if tool_returns:
            newcontent = "\n\n".join([self._str_for_tool_response(tool_return) for tool_return in tool_returns])
            # tool_responses 是 openAI需要用的字段
            response = Msg(name="assistant", role = "tool", tool_responses=tool_returns, content = newcontent)
            LOG_INFO("generate_tool_calls_reply=================end! response:\n",response)
            return True, response

        return False, None

    async def a_generate_tool_calls_reply(
        self,
        messages: Optional[List[Dict]] = None,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[bool, Union[Dict, None]]:
        LOG_INFO("a_generate_tool_calls_reply==================\n")
        """Generate a reply using async function call."""
        if messages is None:
            LOG_INFO("a_generate_tool_calls_reply=================messages none!=\n")
            return False, None
                
        # 获取最近一条消息
        message = messages[-1]
        async_tool_calls = []

        # 如果消息中的 message 键的键值为 空，那么就返回 false 和空
        if not message.get("message"):
            LOG_INFO("a_generate_tool_calls_reply=============message is none or message's value is none!\n")
            return False, None
            
        LOG_INFO("a_generate_tool_calls_reply=================messages:",message)
        tool_message = message.get("message", [])

        # 获取 tool_calls 中的函数调用
        for tool_call in tool_message.get("tool_calls", []):
            LOG_INFO("a_generate_tool_calls_reply=================tool_call:",tool_call)
            # 将异步函数添加到异步函数调用列表中
            async_tool_calls.append(self._a_execute_tool_call(tool_call))
            LOG_INFO("a_generate_tool_calls_reply==================async_tool_calls:",async_tool_calls)

        # 如果有函数调用
        if async_tool_calls:
            # 并发执行函数调用,调用返回的所有的异步函数
            tool_returns = await asyncio.gather(*async_tool_calls)
            newcontent = "\n\n".join([self._str_for_tool_response(tool_return) for tool_return in tool_returns])
            response = Msg(name="assistant", role = "tool", tool_responses=tool_returns, content = newcontent)

            LOG_INFO("a_generate_tool_calls_reply==================end! response:",response)
            return True, response

        return False, None

    def generate_llm_reply(
        self,
        messages: Optional[List[Dict]] = None,
        use_tools: bool = False,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[bool, Union[ModelResponse, None]]:
        """Generate a reply using autogen.oai."""
        LOG_INFO("generate_llm_reply==================begin")

        #=----------------------------------------------
        # 如果模型为空，跳过这个处理则返回 false，和空
        #--------------------------------------------------
        if self.model is None:
            LOG_INFO("generate_llm_reply============self.model is none====go to next reply process!")
            return False, None
        
        # 调用
        # 展开 tool_responses
        all_messages = []
        #--------------------------------------------------
        # 1：首先添加 system prompt
        #--------------------------------------------------
        abc = self._get_sysprompt_msg()
        all_messages.append(abc)
        LOG_INFO("generate_llm_reply======111==_get_sysprompt_msg",all_messages)
        #--------------------------------------------------
        # 2：添加历史记录信息，默认获取 5轮的历史记录
        #--------------------------------------------------
        history =  self._get_history()
        if history:
            all_messages.append(self._get_history())  
        LOG_INFO("generate_llm_reply======112==_get_history",all_messages)
        #--------------------------------------------------
        # 3：添加当前用户消息 messages
        #    - 如果是function call 消息，则本身就是msg消息，只是 tool_responses 是其一个键值的消息。
        #      这个键值消息需要转换成Msg对象，这样在调用大模型的时候可以fomat成 大模型需要传入的 tool_responses格式（openai是一个 tool rsp 就是一个消息--类似msg对象消息--）
        #    - 如果不是function call 消息，则直接添加
        #--------------------------------------------------
        for message in messages:
            # 如果这个消息里面有 tool_responses,说明是一个function call消息，则获取 tool_responses 的内容
            tool_responses = message.get("tool_responses", [])
            # 如果是tool call的消息
            if tool_responses:
                # 如果是 tool消息则转换成Msg对象，并添加到 all_messages 中
                tool_responses_msg = [Msg(name=message.get("name"), role= message.get("role"), content = "", tool_responses = tool_responses)]
                all_messages += tool_responses_msg # 将tool_responses的内容添加到 all_messages 中去
                # 如果当前消息的role不是tool，那么就将除了tool_responses之外的所有键值对添加到 all_messages 中
                if message.get("role") != "tool":
                    # 从字典 message 中提取除 "tool_responses" 键之外的所有键值对，并将它们作为一个新字典添加到 all_messages 列表中
                    other_responses_messsage = {key: message[key] for key in message if key != "tool_responses"}
                    other_responses_msg = Msg(**other_responses_messsage)
                    all_messages.append(other_responses_msg)
            else:
                # 如果不是 tool消息则直接添加
                all_messages.append(message)



        LOG_INFO("generate_llm_reply======113==tool_responses",all_messages)

        #-----------------------------------------------   
        #     *****接收消息的时候保存记忆*****
        # 如果开启了记忆，那么就把接收到的消息存储到记忆中  
        # 正常情况下保存历史记录分成两个部分，一个接收到的消息，一个是大模型返回的消息  
        # 本来都应该在此函数中处理，但是大模型返回的消息可能是流式的，此时返回的response是一个流式生成器 。
        # 而不是实际的内容，流式生成器会返回到用户侧，用户在接受流式消息完成后，整个生成器才算完成，
        # 这个时候才能将大模型返回的内容保存到历史记录中。如果直接再本函数处理了，那么这个内容实际上是未读的生成器内容，
        # 不是实际的内容。保存大模型发回的文本内容，只能放到大模型的处理中保存，流式生成器可以传入agent的保存历史记录函数来保存
        #-----------------------------------------------
        self._save_history(messages) # 保存接收到的消息
        #LOG_INFO("generate_llm_reply=====aaaaaaaaa=113 -2==all_messages",all_messages)
        #==================================
        # 调用 大模型接口获取结果,这里是异步调用
        #==================================
        stream_value = kwargs.get('stream',False) # 获取 stream 参数 
        api_type_value = kwargs.get('api_type',"qwen") # 获取 stream 参数 
        stream_callback = kwargs.get('stream_callback',None) # 获取 stream 参数 
        LOG_INFO("generate_llm_reply======114-1==stream_value",stream_value)
        LOG_INFO("generate_llm_reply======114-2==api_type_value",api_type_value)
        LOG_INFO("generate_llm_reply======114-3==stream_callback",stream_callback)

        LOG_INFO("========================================================")
        LOG_INFO("generate_llm_reply======115-1==all_messages",all_messages)
        LOG_INFO("---------------------------------------------------------")
        LOG_INFO("generate_llm_reply======115-2==messages",messages)
        LOG_INFO("========================================================")

        response = self.model.genrate_rsp(
                                all_messages, 
                                api_type=api_type_value, 
                                stream = stream_value,
                                use_tools = use_tools,
                                stream_callback = stream_callback
                                ) 
       
        #---------------------------------------------------
        #  如果是普通响应, 调用回调函数，保存历史消息
        #  大模型响应回来了，需要保存响应信息为历史记录   
        #  响应都是 ModelResponse 对象格式。在转出去之前，需要转换成Msg对象
        #---------------------------------------------------
        if response.is_funcall == False:
            llm_rsp = Msg(name="assistant",role="assistant",content=response.text)
        else:
            llm_rsp = Msg(name="assistant",role="tool",content="",message = response.text)

        #==================================    
        # 保存历史记录
        #==================================  
        self._save_history(llm_rsp) # 保存发出去的消息内容

        LOG_INFO("generate_llm_reply==================end rsp:###############################\n",llm_rsp)
        # 如果有 response则返回 true，和结果，否则返回 false，和空
        return (False, None) if llm_rsp is None else (True, llm_rsp)

    # 这个是 register_reply 注册到 回复里面的函数 
    async def a_generate_llm_reply(
        self,
        messages: Optional[List[Dict]] = None,
        use_tools: bool = False,
        config: Optional[Any] = None,
        **kwargs: Any
    ) -> Tuple[bool, Union[ModelResponse, None]]:
        """Generate a reply using autogen.oai."""
        LOG_INFO("a_generate_llm_reply==================begin")

        #=----------------------------------------------
        # 如果模型为空，跳过这个处理则返回 false，和空
        #--------------------------------------------------
        if self.model is None:
            LOG_INFO("a_generate_llm_reply============self.model is none====go to next reply process!")
            return False, None
        
        # 调用
        # 展开 tool_responses
        all_messages = []
        #--------------------------------------------------
        # 1：首先添加 system prompt
        #--------------------------------------------------
        abc = self._get_sysprompt_msg()
        all_messages.append(abc)

        #--------------------------------------------------
        # 2：添加历史记录信息，默认获取 5轮的历史记录
        #--------------------------------------------------
        all_messages.append(self._get_history())  

        #--------------------------------------------------
        # 3：添加当前用户消息
        #--------------------------------------------------
        for message in messages:
            tool_responses = message.get("tool_responses", [])
            if tool_responses:
                # 如果是 tool消息则转换成Msg对象，并添加到 all_messages 中
                tool_responses_msg = [Msg(name=message.get("name"), role= message.get("role"), content = "", tool_responses = tool_responses)]
                all_messages += tool_responses_msg
                # tool role on the parent message means the content is just concatenation of all of the tool_responses
                if message.get("role") != "tool":
                    # 从字典 message 中提取除 "tool_responses" 键之外的所有键值对，并将它们作为一个新字典添加到 all_messages 列表中
                    # 从字典 message 中提取除 "tool_responses" 键之外的所有键值对，并将它们作为一个新字典添加到 all_messages 列表中
                    other_responses_messsage = {key: message[key] for key in message if key != "tool_responses"}
                    other_responses_msg = Msg(**other_responses_messsage)
                    all_messages.append(other_responses_msg)
            else:
                all_messages.append(message)

        #-----------------------------------------------   
        #     *****接收消息的时候保存记忆*****
        # 如果开启了记忆，那么就把接收到的消息存储到记忆中
        #-----------------------------------------------
        self._save_history(messages)

        #==================================
        # 调用 大模型接口获取结果,这里是异步调用
        #==================================
        stream_value = kwargs.get('stream', False) # 获取 stream 参数 
        api_type_value = kwargs.get('api_type',"qwen") # 获取 stream 参数 
        stream_callback = kwargs.get('stream_callback',None) # 获取 stream 参数 
        LOG_INFO("a_generate_llm_reply======114-1==stream_value",stream_value)
        LOG_INFO("a_generate_llm_reply======114-2==api_type_value",api_type_value)
        LOG_INFO("a_generate_llm_reply======114-3==stream_callback",stream_callback)

        response = await self.model.a_genrate_rsp(
                                        all_messages, 
                                        api_type=api_type_value, 
                                        stream = stream_value,
                                        use_tools = use_tools,
                                        stream_callback = stream_callback
                                        ) 
       
        #---------------------------------------------------
        #  如果是普通响应, 调用回调函数，保存历史消息
        #  大模型响应回来了，需要保存响应信息为历史记录   
        #  响应都是 ModelResponse 对象格式。在转出去之前，需要转换成Msg对象
        #---------------------------------------------------
        if response.is_funcall == False:
            llm_rsp = Msg(name="assistant",role="assistant",content=response.text)
        else:
            llm_rsp = Msg(name="assistant",role="tool",content="",message = response.text)

        #==================================    
        # 保存历史记录
        #==================================  
        self._save_history(llm_rsp) # 保存发出去的消息内容

        LOG_INFO("a_generate_llm_reply==================end\n rsp:",llm_rsp) 
        # 如果 extracted_response 解析出来的结果为 空则返回 false，结束，否则 返回 true，和结果
        return (False, None) if llm_rsp is None else (True, llm_rsp)

    #从命令行获取输入
    def get_input_from_command_line(prompt: str) -> str:
        return input(prompt)

    #从消息获取输入
    def get_input_from_message(prompt: str) -> str:
        LOG_INFO(prompt)
        # 模拟等待用户消息的过程
        time.sleep(5)  # 假设等待5秒钟
        return "用户的消息"

    # 同步函数
    def get_human_input(self, prompt: str) -> str:

        reply = self.human_io_handler.input(prompt)
        self._human_input.append(reply)
        return reply

    # 异步函数
    async def a_get_human_input(self, prompt: str) -> str:
        """(Async) Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        """
        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(None, functools.partial(self.get_human_input, prompt))
        return reply

    # 同步函数，将内容输出给 用户
    def output_to_human(self, prompt: str) -> str:
        self.human_io_handler.output(prompt)

    # 异步函数
    async def a_output_to_human(self, prompt: str) -> str:
        """(Async) Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        """
        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(None, functools.partial(self.output_to_human, prompt))
        return reply

    # 向Agent注册一个工具，入参是 工具的函数名，工具的名称，工具的描述
    def register_tool_for_llm(self, func: Callable, name: str = None, description: str = None):
        """
        注册一个工具到agent中的tool_manager工具管理器中。

        :func: 工具的函数名。
        :name: 工具的名称。         - 可不填，默认为函数名。
        :description: 工具的描述。  - 可不填，默认为空。
        """
        # 创建一个新的工具实例
        new_agent_tool = AgentTool(func, name=name, description=description)

        #将工具注册到工具管理器中
        self.tool_manager.register_tool(new_agent_tool)

    def is_agent_tool_exist(self, tool_name: str) -> bool:
        """
        调用通过代理注册的工具。

        :param tool_name: 要调用的工具名称。
        :param params: 传递给工具的参数字典。
        :return: 工具执行后的返回值。
        """
        return self.tool_manager.is_tool_exist(tool_name)
    
    # agent调用工具，传入工具的名称和工具的参数
    def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        调用通过代理注册的工具。

        :param tool_name: 要调用的工具名称。
        :param params: 传递给工具的参数字典。
        :return: 工具执行后的返回值。
        """
        return self.tool_manager.call_tool(tool_name, params)

    # 获取工具的描述信息，传入的参数是工具的类型，默认是openai
    def get_llm_tool_descriptions(self, tool_type:Literal["openai", "qwen"] = "openai") -> List[Dict[str, Any]]:
        """
        获取tools的描述信息。

        """
        return self.tool_manager.get_tool_descriptions(tool_type)
    
    # 从消息里面解析出工具的名称和参数
    def _parse_message_for_tool(self, message: str) -> Any:
        """
        解析消息以提取工具名称和参数。

        :param message: 用户输入的消息。
        :return: 工具名称和参数字典。
        """
        # 这里是一个简单的示例解析
        #return "get_delivery_date", {"order_id": "12345"}
        pass