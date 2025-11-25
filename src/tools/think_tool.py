"""
Think Tool for AI Reflection
让 AI 在关键步骤后进行思考和反思的工具
"""

from langchain.tools import tool


@tool
def think(thought: str) -> str:
    """
    使用此工具来思考当前的情况并规划下一步行动。

    Returns:
        确认消息，鼓励继续进行
    """
    word_count = len(thought.split())
    return (
        f"思考已记录（{word_count} 字）。"
        f"你的反思有助于确保一个深思熟虑的方法。"
        f"根据这个分析继续你的下一步行动。"
    )


__all__ = ["think"]
