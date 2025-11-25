from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAgent(ABC):
    """Agent 和 MutiAgent 的抽象基类

    所有子类必须实现以下四个方法:
    - _generate_tools: 生成工具列表
    - _generate_middleware: 生成中间件列表
    - _call_llm: 调用 LLM 生成响应
    - _extract_messages: 提取/构造消息列表
    """

    @abstractmethod
    def _generate_tools(self) -> List[Any]:
        """生成 langchain 的 tools 列表

        Returns:
            工具列表，用于 Agent 执行任务
        """
        pass

    @abstractmethod
    def _generate_middleware(self, **kwargs) -> List[Any]:
        """生成 langchain 的中间件列表

        Args:
            **kwargs: 额外参数，如 llm 实例等

        Returns:
            中间件列表，用于处理对话历史、重试等
        """
        pass

    @abstractmethod
    def _generate_agent(self, **kwargs) -> List[Any]:
        """构建一个由多个工具节点组成的 LangGraph workflow。"""
        pass

    @abstractmethod
    async def _call_llm(
        self, messages: List[Any], thread_id: str = "default", **kwargs
    ) -> Dict[str, Any] | Any:
        """调用 LLM 生成响应

        Args:
            messages: 消息列表
            thread_id: 对话线程 ID
            **kwargs: 额外参数

        Returns:
            LLM 的响应结果
        """
        pass

    @abstractmethod
    def _extract_messages(self, **kwargs) -> List[Any]:
        """从输入参数中提取或构造消息列表，供 LLM 调用使用。

        Args:
            **kwargs: 可能包含历史对话、用户输入或工具输出等上下文信息

        Returns:
            消息列表（如 HumanMessage/AIMessage 等），按顺序排列
        """
        pass
