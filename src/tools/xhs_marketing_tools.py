"""
Xiaohongshu Marketing Tools
小红书夸张营销文案工具：为大模型提供爆款口吻、标题和文案模板。
"""

from typing import List, Optional

from langchain.tools import tool


@tool(return_direct=True)
def xhs_style_guide(
    persona: str = "疯批反差萌",
    emoji_density: int = 3,
    safety_note: bool = True,
) -> str:
    """
    返回一份夸张小红书写作指南，直接贴给大模型即可套用。

    Args:
        persona: 口吻人设，如“疯批反差萌”“专业又尖叫”
        emoji_density: 每段建议插入的 emoji 数量
        safety_note: 是否提醒规避医疗/功效绝对化表述

    Returns:
        一段包含语气、结构、标点、标签用法的写作速查表
    """
    guide = [
        f"人设：{persona}，抓马到位但保持真情实感；对读者称呼用“姐妹们/宝子们”。",
        f"语气：开头必须惊叫 + 反复感叹；多用大写和拉长词（太！！！好！！！哭了！！！）。",
        "结构：爆点开头 -> 个人崩溃瞬间/反转 -> 3-5 个细节卖点 -> 强制安利 + 行动口号。",
        f"标点：感叹号连发，省略号制造悬念；每段插入约 {emoji_density} 个 emoji（⚡️🤯😭✨🫶🔥）。",
        "标签：结尾叠加 6-10 个话题标签，包含产品、场景、情绪、趋势关键词。",
    ]
    if safety_note:
        guide.append("合规：避免“治愈/百分百”之类绝对功效词，可用“离谱好用”“像开挂”。")

    return "\n".join(f"- {line}" for line in guide)


@tool(return_direct=True)
def xhs_title_pack(
    product: str,
    target_user: str = "姐妹们",
    scene: Optional[str] = None,
) -> str:
    """
    生成一组高点击小红书风格标题，直接可用。

    Args:
        product: 产品/服务名称
        target_user: 主要受众称呼
        scene: 使用场景或痛点

    Returns:
        6-8 条标题候选，带 emoji 和话题位
    """
    scene_part = f"{scene} " if scene else ""
    titles: List[str] = [
        f"{target_user}崩溃尖叫！{scene_part}{product}真的离谱好用！！！⚡️🤯",
        f"跪了！{product}=开挂神器？我试完沉默了😭",
        f"【别再错过】{scene_part}{product}这波我必须全网喊！！！🔥",
        f"没有对比没有伤害，{product}把我拿捏了…🫠🫶",
        f"年度心动榜第一名：{product}！把状态拉满的一天✨",
        f"冲！{product} = 我最勇敢的一次入手，结果直接上头😳",
        f"反转了！原来{scene_part}{product}才是隐藏王者？！🤯",
    ]
    hashtags = [
        f"#{product}",
        f"# {scene}" if scene else "",
        "# 必入好物",
        "# 尖叫推荐",
        "# 小众宝藏",
    ]
    hashtag_line = " ".join(tag for tag in hashtags if tag)
    titles.append(f"标签备选：{hashtag_line}")
    return "\n".join(f"{idx+1}. {title}" for idx, title in enumerate(titles))


@tool(return_direct=True)
def xhs_hype_copy(
    product: str,
    selling_points: str,
    audience: str = "姐妹们",
    scenario: str = "日常通勤",
    call_to_action: str = "冲！马上安排！",
) -> str:
    """
    生成一篇夸张的小红书种草文案，含开头爆点、细节卖点和标签。

    Args:
        product: 产品/服务名称
        selling_points: 卖点列表，用逗号分隔
        audience: 读者称呼
        scenario: 使用场景
        call_to_action: 行动口号

    Returns:
        一段完整可直接发布的夸张文案
    """
    points = [
        p.strip() for p in selling_points.replace("，", ",").split(",") if p.strip()
    ]
    point_lines = "\n".join(f"· {idx+1}）{p} ✅" for idx, p in enumerate(points))
    if not point_lines:
        point_lines = "· 太多亮点了根本写不完，自己感受！！"

    hashtags = [
        f"#{product}",
        f"# {scenario}",
        "# 必入好物",
        "# 拯救打工人",
        "# 种草不踩雷",
        "# 爆改生活",
    ]
    header = (
        f"{audience}！！！我直接破防！{scenario}被{product}狠狠拿捏，太炸裂了😭😭😭"
    )
    story = (
        f"本来只想随便试试，结果一上手就像开挂，离谱到想冲进评论区喊停！"
        f" 细节我掰开揉碎告诉你："
    )
    cta = f"{call_to_action} 不冲真的会后悔一整年！"

    return "\n".join(
        [
            header,
            "—" * 10,
            story,
            point_lines,
            "—" * 10,
            cta,
            " ".join(hashtags),
        ]
    )


__all__ = ["xhs_style_guide", "xhs_title_pack", "xhs_hype_copy"]
