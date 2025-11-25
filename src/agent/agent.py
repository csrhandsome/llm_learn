"""
Underwater VLM Agent
水下视觉语言模型 Agent，用于控制水下机器人（ROV）
"""

import asyncio
import datetime
from typing import Any, Dict, List

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, ToolMessage

from src.agent.base_agent import BaseAgent
from src.config import VLM_CONFIG
from src.prompts.prompt import SYSTEM_PROMPT


class Agent(BaseAgent):
    def __init__(self):
        # 从配置文件读取 VLM 参数
        self.llm = ChatOpenAI(
            model=VLM_CONFIG["model_name"],
            api_key=VLM_CONFIG["api_key"],
            base_url=VLM_CONFIG["base_url"],
            temperature=0.0,
        )

        # 创建 Agent
        self.agent = self._generate_agent()

    def _generate_tools(self) -> List[Any]:
        """
        控制工具列表

        Returns:
            工具列表
        """
        return []

    def _generate_agent(self, **kwargs):
        agent = create_agent(
            self.llm,
            system_prompt=SYSTEM_PROMPT,
            tools=self._generate_tools(),
            middleware=self._generate_middleware(),
            checkpointer=InMemorySaver(),
        )
        return agent

    def _generate_middleware(self, **kwargs) -> List[Any]:
        """
        生成中间件列表

        当前不使用额外中间件，返回空列表

        Returns:
            空列表
        """
        return []

    async def _call_llm(
        self, messages: List[Any], thread_id: str = "default", **kwargs
    ) -> Dict[str, Any] | Any:
        """
        调用 LLM 生成响应

        Args:
            messages: 消息列表（可包含图像）
            thread_id: 对话线程 ID
            **kwargs: 额外参数

        Returns:
            LLM 的响应结果
        """
        try:
            result = await self.agent.ainvoke(
                {"messages": messages},
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "max_iterations": 10,
                    },
                    "recursion_limit": 20,
                },
            )
            return result
        except Exception as e:
            return {"error": str(e), "messages": []}

    def _extract_messages(self, **kwargs) -> List[Any]:
        """
        从输入参数中提取消息列表

        Args:
            **kwargs: 可能包含 'messages', 'command', 'image_list' 等

        Returns:
            消息列表
        """
        messages = kwargs.get("messages")
        if isinstance(messages, list):
            return messages

        # 如果只有命令文本
        command = kwargs.get("command", "")
        if command:
            return [HumanMessage(content=command)]

        return []

    async def process_command(
        self,
        command: str,
        thread_id: str = "default",
    ) -> Dict[str, Any]:
        """
        处理用户的指令

        Args:
            command: 用户的自然语言指令
            thread_id: 对话线程ID

        Returns:
            包含响应和控制数据的字典
        """
        pass


# 别名，保持向后兼容
ControlAgent = Agent
Agent = Agent


async def main():
    """测试主函数"""
    agent = Agent()


if __name__ == "__main__":
    asyncio.run(main())
