import os
import json
import asyncio
import time
import datetime
import operator
from typing import Any, Annotated, Dict, List, Optional, Set, TYPE_CHECKING, TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ClearToolUsesEdit,
    ToolRetryMiddleware,
    SummarizationMiddleware,
)
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from pathlib import Path
import tqdm

from src.agent.prompt import (
    SYSTEM_PROMPT,
    USER_INPUT_TEMPLATE,
    ERROR_FEEDBACK_TEMPLATE,
    EMPTY_RESULT_FEEDBACK_TEMPLATE,
    RESULT_VERIFICATION_TEMPLATE,
)
from src.util import load_questions, safe_read_json_with_lock, safe_write_json_with_lock
from src.database.sql_exe import execute_sql_with_pymysql
from src.config import LLM_CONFIG
from src.config import DB_CONFIG
from src.tool import (
    # 阶段性工具集
    SCHEMA_PLANNING_TOOLS,
    EXEMPLAR_TOOLS,
    SQL_GENERATION_TOOLS,
    VALIDATION_EXECUTION_TOOLS,
    # 完整工具集
    ALL_TOOLS,
    CORE_TOOLS,
    ADVANCED_TOOLS,
)
from agent.base_agent import BaseAgent


class AgentState(TypedDict):
    """State shared across LangGraph nodes."""

    messages: Annotated[List[BaseMessage], operator.add]


class MutiAgent(BaseAgent):
    def __init__(
        self,
        model_name: Optional[
            str
        ] = None,  # 默认模型，可选: "deepseek-ai/DeepSeek-V3.2-Exp", "MiniMaxAI/MiniMax-M2", "zai-org/GLM-4.6"
    ):
        self.model_name = model_name or LLM_CONFIG["model_name"]
        self.llm = self._generate_llm(self.model_name)
        # agnet里面内置了AgentState，在每次invoke的时候可以保存记忆
        self.agent = self._generate_agent()
        self.sql_executor = execute_sql_with_pymysql()
        self.all_questions = load_questions(Path("data/final_dataset.json"))

    def _generate_tools(self) -> List[Any]:
        """生成默认工具列表

        MutiAgent 使用分阶段的工具集，此方法返回所有可用工具。
        实际使用时通过 _build_stage_agent 为每个阶段指定不同的工具。

        Returns:
            所有可用工具的列表
        """
        return ALL_TOOLS

    def _generate_llm(self, model_name: str = None) -> ChatOpenAI:
        """生成 LLM 实例，支持多种模型

        支持的模型:
        1. deepseek-ai/DeepSeek-V3.2-Exp: DeepSeek 实验性模型，使用 DSA 稀疏注意力机制
        2. MiniMaxAI/MiniMax-M2: 紧凑高效的 MoE 模型，230B 总参数，10B 激活参数
        3. zai-org/GLM-4.6: GLM-4.6，200K 上下文窗口，强大的工具使用能力

        Args:
            model_name: 模型名称，如果为 None 则使用配置文件中的模型

        Returns:
            ChatOpenAI 实例
        """
        if model_name is None:
            model_name = LLM_CONFIG["model_name"]

        return ChatOpenAI(
            model=model_name,
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            temperature=0.0,
        )

    def _build_stage_agent(self, *, llm: ChatOpenAI, tools: List[Any]):
        """Helper to build a stage-specific LangGraph agent."""

        return create_agent(
            llm,
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            middleware=self._generate_middleware(llm=llm),
        )

    def _generate_agent(self):
        """构建一个由多个工具节点组成的 LangGraph workflow。

        每个阶段使用不同的模型以优化性能和成本:
        1. Schema 分析阶段: 使用 GLM-4.6 (200K 上下文，适合理解复杂 schema)
        2. 示例检索阶段: 使用 MiniMax-M2 (快速高效，适合检索任务)
        3. SQL 生成阶段: 使用 DeepSeek-V3.2-Exp (强大的代码生成能力)
        4. 验证执行阶段: 使用 GLM-4.6 (强工具使用能力，适合调试)
        """
        # 这一块修改一下，现在是模仿agent.py里面的操作，第一个用"zai-org/GLM-4.6"来完成现在的前三个阶段的工作。然后用"zai-org/GLM-4.6"和MiniMax-M2来评审这个任务，然后综合一个输出。常住一个deepseek的agent,如果有错误就认真分析错误，如果没有错误就跳过

        # 阶段 1: Schema 分析 - 使用 GLM-4.6 (200K 上下文窗口，适合理解复杂 schema)
        schema_llm = self._generate_llm("zai-org/GLM-4.6")
        schema_agent = self._build_stage_agent(
            llm=schema_llm, tools=SCHEMA_PLANNING_TOOLS
        )

        # 阶段 2: 示例检索 - 使用 MiniMax-M2 (紧凑高效，10B 激活参数)
        exemplar_llm = self._generate_llm("MiniMaxAI/MiniMax-M2")
        exemplar_agent = self._build_stage_agent(llm=exemplar_llm, tools=EXEMPLAR_TOOLS)

        # 阶段 3: SQL 生成 - 使用 DeepSeek-V3.2-Exp (强大的代码生成和推理能力)
        generation_llm = self._generate_llm("deepseek-ai/DeepSeek-V3.2-Exp")
        generation_agent = self._build_stage_agent(
            llm=generation_llm, tools=SQL_GENERATION_TOOLS
        )

        # 阶段 4: 验证执行 - 使用 GLM-4.6 (强工具使用能力，适合调试和验证)
        validation_llm = self._generate_llm("zai-org/GLM-4.6")
        validation_agent = self._build_stage_agent(
            llm=validation_llm,
            tools=VALIDATION_EXECUTION_TOOLS,
        )

        # 构建 StateGraph workflow
        # create_agent 返回的 agent 本身就是一个可调用的节点，不需要额外包装
        builder = StateGraph(AgentState)
        builder.add_node("schema_analysis", schema_agent)
        builder.add_node("example_retrieval", exemplar_agent)
        builder.add_node("sql_generation", generation_agent)
        builder.add_node("sql_validation", validation_agent)

        # 定义节点间的流转顺序
        builder.add_edge(START, "schema_analysis")
        builder.add_edge("schema_analysis", "example_retrieval")
        builder.add_edge("example_retrieval", "sql_generation")
        builder.add_edge("sql_generation", "sql_validation")
        builder.add_edge("sql_validation", END)

        return builder.compile(checkpointer=InMemorySaver())

    async def _call_llm(
        self, messages: List[Any], thread_id: str = "default"
    ) -> dict[str, Any] | Any:
        """调用agent,agent根据tools列表来判断是否使用工具并执行,返回响应。

        Args:
            messages: 当前轮次的新消息列表,LangChain会自动通过thread_id加载历史消息
            thread_id: 对话线程ID,用于隔离不同问题的对话历史。每个问题应使用唯一的thread_id

        """

        # LangChain通过checkpointer和thread_id自动管理对话历史
        # 每个问题使用独立的thread_id，避免历史在不同问题间累积
        result = await self.agent.ainvoke(
            {
                "messages": messages,
            },
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "max_iterations": 150,  # 最多15次迭代（工具调用）防止无限循环
                },
                "recursion_limit": 100,  # 降低递归限制到50，更早发现死循环问题
            },  # 使用传入的thread_id，为每个问题创建独立的对话历史
            # context=context,  # 传递context给agent
        )

        return result

    def _generate_middleware(self, llm: Optional[ChatOpenAI] = None):
        """langchain的中间件

        中间件列表:
        1. SummarizationMiddleware: 当对话历史过长时自动总结，避免 token 超限
        2. ToolRetryMiddleware: 工具调用失败时自动重试
        """
        llm = llm or self.llm
        return [
            # 摘要中间件 - 防止对话历史过长导致 token 超限
            SummarizationMiddleware(
                model=llm,  # 使用经济型模型进行摘要
                max_tokens_before_summary=140000,  # 对话达到 100k tokens 时触发摘要
                messages_to_keep=15,  # 摘要后保留最近 15 条消息
                summary_prompt="""请总结以下对话历史，保留关键信息:
1. 用户的原始问题和需求
2. 已识别的数据库表和列
3. SQL 生成过程中的重要决策
4. 遇到的错误和修复方法
5. 当前的 SQL 查询状态
6. 调用了哪些工具以及它们的输出
总结应该简洁但包含所有必要的上下文，以便继续对话。""",
            ),
            # 工具重试中间件 - 工具调用失败时自动重试
            ToolRetryMiddleware(
                max_retries=3,  # 最多重试 3 次
                backoff_factor=2.0,  # 指数退避倍数
                initial_delay=1.0,  # 初始延迟 1 秒
                max_delay=60.0,  # 最大延迟 60 秒
                jitter=True,  # 添加随机抖动避免雷鸣羊群效应
            ),
        ]

    def _append_assistant_tool_message(
        self, messages: List[Any], assistant_message
    ) -> None:
        """将包含tool_calls的assistant消息追加到对话历史。"""
        if assistant_message is None:
            return

        message_obj = assistant_message

        if isinstance(message_obj, tuple):
            for item in message_obj:
                if hasattr(item, "content") or isinstance(item, dict):
                    message_obj = item
                    break
        elif isinstance(message_obj, list):
            for item in message_obj:
                if hasattr(item, "content") or isinstance(item, dict):
                    message_obj = item
                    break

        if isinstance(message_obj, dict):
            content = message_obj.get("content", "") or ""
            tool_calls = message_obj.get("tool_calls") or []
        else:
            content = getattr(message_obj, "content", "") or ""
            tool_calls = getattr(message_obj, "tool_calls", None)

        if tool_calls:
            formatted_calls: List[Dict[str, Any]] = []
            for tool_call in tool_calls:
                tool_name = tool_call.get("name")
                tool_id = tool_call.get("id")
                tool_type = tool_call.get("type")
                tool_args = tool_call.get("args", {}) or {}
                formatted_calls.append(
                    {
                        "id": tool_id,
                        "type": tool_type,
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False),
                        },
                    }
                )
            # 使用 AIMessage 并传递 tool_calls
            ai_message = AIMessage(
                content=content, additional_kwargs={"tool_calls": formatted_calls}
            )
            messages.append(ai_message)
        else:
            messages.append(AIMessage(content=content))

    def _extract_messages(self, **kwargs) -> List[Any]:
        """从 kwargs 中提取消息列表，默认返回空列表。"""
        messages = kwargs.get("messages")
        return messages if isinstance(messages, list) else []

    def _generate_user_text(
        self, query: str, table_list: List[str], knowledge: str
    ) -> str:
        """生成用户输入的文本，使用模板格式化"""
        # 格式化 table_list
        formatted_tables = ", ".join(table_list)
        # 格式化 knowledge
        formatted_knowledge = knowledge if knowledge else "None"

        # 使用模板
        user_text = USER_INPUT_TEMPLATE.format(
            query=query, table_list=formatted_tables, knowledge=formatted_knowledge
        )
        return user_text

    def _extract_sql_from_response(self, response: str) -> str:
        """从 LLM 响应中提取 SQL 语句

        Args:
            response: LLM 的完整响应

        Returns:
            提取的 SQL 字符串
        """
        # 尝试从 ```sql ... ``` 中提取
        if "```sql" in response:
            sql = response.split("```sql")[-1].split("```")[0].strip()
            return sql
        # 如果没有代码块，尝试找 SELECT 开头的语句
        elif "SELECT" in response.upper():
            lines = response.split("\n")
            sql_lines = []
            in_sql = False
            for line in lines:
                if "SELECT" in line.upper():
                    in_sql = True
                if in_sql:
                    sql_lines.append(line)
                    # 如果遇到分号，结束
                    if ";" in line:
                        break
            return "\n".join(sql_lines).strip()
        else:
            print(f"Error extracting SQL")
            return response.strip()

    async def solve_question(
        self, question_data: Dict[str, str], messages: List[Any], attempt: int = 0
    ) -> Dict[str, str]:
        """解决一个问题，生成并执行 SQL

        此方法会：
        1. 调用 Agent 生成 SQL
        2. 执行 SQL 并返回结果
        3. 如果失败，返回错误信息（由 main 函数决定是否重试）

        Args:
            question_data: 包含 question, table_list, knowledge, sql_id 的字典
            messages: 对话历史消息列表
            attempt: 当前尝试次数（用于生成 thread_id）

        Returns:
            Dict: 包含 sql_id, sql, status, result/error_message 的字典
        """
        sql_id = question_data.get("sql_id", "")
        # 调用 LLM 生成 SQL
        # 使用 sql_id_attempt 作为 thread_id，每次尝试使用不同的 thread_id
        # attempt_id = f"{sql_id}_{attempt}"
        attempt_id = sql_id
        result = await self._call_llm(messages, thread_id=attempt_id)

        # 从结果中提取 SQL
        if "messages" in result:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                response_text = last_message.content
            else:
                response_text = str(last_message)
        else:
            response_text = str(result)

        # 解析 SQL
        sql = self._extract_sql_from_response(response_text)

        # 执行 SQL 验证
        exec_result = self.sql_executor.execute_single_sql(sql, DB_CONFIG)

        # 返回结果（包含执行状态）
        return {
            "sql_id": sql_id,
            "sql": sql,
            "status": exec_result["status"],
            "result": exec_result.get("result"),
            "error_message": exec_result.get("error_message"),
        }


async def main():
    """主函数：批量处理问题，生成并验证 SQL"""
    # 定义输入和输出文件路径
    input_filepath = Path("data/final_dataset.json")
    generated_sqls_path = Path("generated_sqls.json")  # 临时文件
    final_output_filepath = Path("dataset_exe_result_mutiagent.json")  # 最终提交文件
    max_retries = 12  # 最大重试次数

    all_questions = load_questions(input_filepath)
    agent = MutiAgent()

    # 如果文件已存在，加载已完成的结果（使用文件锁）
    completed_ids = set()
    results_list = safe_read_json_with_lock(final_output_filepath)
    if results_list:
        completed_ids = {item["sql_id"] for item in results_list}
        print(f"从检查点恢复，已完成 {len(completed_ids)} 个任务")
    else:
        print("检查点文件为空或不存在，从头开始")

    count = 0
    # 生成答案（带自动重试和验证）
    for question_data in tqdm.tqdm(all_questions, desc="Agent 正在生成 SQL"):
        query = question_data.get("question", "")
        table_list = question_data.get("table_list", [])
        knowledge = question_data.get("knowledge", "")
        sql_id = question_data.get("sql_id", "")
        golden_sql = question_data.get("golden_sql", "")

        count += 1
        last_check = False
        # if count < 95:
        #     continue

        # 跳过已完成的任务
        if sql_id in completed_ids:
            print(f"⏭ {sql_id} 已完成，跳过")
            continue
        # 如果是golden_sql，直接执行已有SQL
        if golden_sql:
            exec_result = agent.sql_executor.execute_single_sql(
                question_data.get("sql", ""), DB_CONFIG
            )
            final_result = {
                "sql_id": sql_id,
                "sql": question_data.get("sql", ""),
                "status": exec_result["status"],
                "result": exec_result.get("result"),
            }
            print(f"✓ {sql_id} 使用 Golden SQL (跳过 Agent)")

            # 使用文件锁安全写入
            results_list = safe_write_json_with_lock(
                final_output_filepath, final_result
            )
            completed_ids.add(sql_id)
            continue
        # 使用模板生成用户输入
        user_text = agent._generate_user_text(query, table_list, knowledge)

        # 初始化消息列表
        messages = [HumanMessage(content=user_text)]

        # 重试循环
        final_result = None
        for attempt in range(max_retries):
            # 调用 solve_question 生成并执行 SQL
            result = await agent.solve_question(question_data, messages, attempt)

            # 如果执行成功，检查结果是否为空
            if result["status"] == "success":
                result_data = result.get("result", [])

                # 检查结果是否为空
                if not result_data or len(result_data) == 0:
                    # 结果为空，要求 Agent 检查质量
                    if attempt < max_retries - 1:
                        print(
                            f"{sql_id} SQL 执行成功但结果为空 (尝试 {attempt + 1}/{max_retries})"
                        )

                        # 使用模板构建反馈消息
                        empty_result_feedback = EMPTY_RESULT_FEEDBACK_TEMPLATE.format(
                            sql=result["sql"]
                        )
                        messages.append(HumanMessage(content=empty_result_feedback))
                        continue
                    else:
                        # 不为空，是成功的

                        # 如果还没有进行最终检查，且还有重试机会，让 LLM 验证结果
                        if not last_check and attempt < max_retries - 4:
                            last_check = True
                            print(f"🔍 {sql_id} 将执行结果传入 LLM 进行最终验证")

                            # 构建结果验证消息
                            result_preview = (
                                result_data[:5] if len(result_data) > 5 else result_data
                            )
                            verification_message = RESULT_VERIFICATION_TEMPLATE.format(
                                sql=result["sql"],
                                result_preview=result_preview,
                                result_count=len(result_data),
                                original_question=query,
                            )
                            messages.append(HumanMessage(content=verification_message))
                            continue  # 继续循环，让 LLM 有机会优化

                        # 最终检查后，或没有最终检查机会，保存结果
                        final_result = {
                            "sql_id": sql_id,
                            "sql": result["sql"],
                            "status": "success",
                            "result": result_data,
                        }
                        break

            # 如果执行失败且还有重试机会，将错误信息反馈给 Agent
            if attempt < max_retries - 1:
                error_msg = result["error_message"]
                print(
                    f"✗ {sql_id} SQL 执行失败 (尝试 {attempt + 1}/{max_retries}): {error_msg[:100]}..."
                )

                # 使用模板构建错误反馈消息
                feedback_text = ERROR_FEEDBACK_TEMPLATE.format(
                    sql=result["sql"], error_message=error_msg
                )
                messages.append(HumanMessage(content=feedback_text))
            else:
                # 已达到最大重试次数，保存最后的错误
                print(f"✗ {sql_id} SQL 执行失败，已达到最大重试次数 ({max_retries})")
                final_result = {
                    "sql_id": sql_id,
                    "sql": result["sql"],
                    "status": "error",
                    "error_message": result["error_message"],
                }

        # 添加最终结果到列表
        if final_result:
            # 使用文件锁安全写入（实时追加）
            results_list = safe_write_json_with_lock(
                final_output_filepath, final_result
            )
            completed_ids.add(sql_id)

    # 最终结果已经通过文件锁实时保存，这里不需要再次保存
    print(f"\n✅ 所有任务已完成，结果已保存到 {final_output_filepath}")

    # 统计结果
    success_count = sum(1 for r in results_list if r["status"] == "success")
    error_count = len(results_list) - success_count
    print(f"成功: {success_count}, 失败: {error_count}")


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
