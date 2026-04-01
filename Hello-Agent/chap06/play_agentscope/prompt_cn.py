# -*- coding: utf-8 -*-
"""三国狼人杀中文提示词"""
from typing import Optional, List, Any
from structured_output_cn import (
    DiscussionModelCN,
    get_vote_model_cn,
    WitchActionModelCN,
    get_seer_model_cn,
    get_hunter_model_cn,
    WerewolfKillModelCN
)
from json_utils import get_json_prompt_instructions


class ChinesePrompts:
    """中文提示词管理类"""
    
    @staticmethod
    def get_role_prompt(role: str, character: str) -> str:
        """获取角色提示词"""
        base_prompt = f"""你是{character}，在这场三国狼人杀游戏中扮演{role}。

{get_json_prompt_instructions(DiscussionModelCN)}

角色特点：
"""
        
        if role == "狼人":
            return base_prompt + f"""
- 你是狼人阵营，目标是消灭所有好人
- 夜晚可以与其他狼人协商击杀目标
- 白天要隐藏身份，误导好人
- 以{character}的性格说话和行动
"""
        elif role == "预言家":
            return base_prompt + f"""
- 你是好人阵营的预言家，目标是找出所有狼人
- 每晚可以查验一名玩家的真实身份
- 要合理公布查验结果，引导好人投票
- 以{character}的智慧和洞察力分析局势
"""
        elif role == "女巫":
            return base_prompt + f"""
- 你是好人阵营的女巫，拥有解药和毒药各一瓶
- 解药可以救活被狼人击杀的玩家
- 毒药可以毒杀一名玩家
- 要谨慎使用道具，在关键时刻发挥作用
"""
        elif role == "猎人":
            return base_prompt + f"""
- 你是好人阵营的猎人
- 被投票出局时可以开枪带走一名玩家
- 要在关键时刻使用技能，带走狼人
- 以{character}的勇猛和决断力行动
"""
        else:  # 村民
            return base_prompt + f"""
- 你是好人阵营的村民
- 没有特殊技能，只能通过推理和投票
- 要仔细观察，找出狼人的破绽
- 以{character}的性格参与讨论
"""
    
    @staticmethod
    def get_kill_prompt(players: List[Any]) -> str:
        """获取狼人击杀提示"""
        return f"""请选择今晚要击杀的目标。

{get_json_prompt_instructions(WerewolfKillModelCN)}

可选目标：{', '.join([p.name for p in players])}
"""
    
    @staticmethod
    def get_seer_prompt(players: List[Any]) -> str:
        """获取预言家查验提示"""
        return f"""请选择今晚要查验的玩家。

{get_json_prompt_instructions(get_seer_model_cn(players))}

可选目标：{', '.join([p.name for p in players])}
"""
    
    @staticmethod
    def get_witch_prompt(killed_player: Optional[str]) -> str:
        """获取女巫行动提示"""
        death_info = f"今晚{killed_player}被狼人击杀" if killed_player else "今晚平安无事"
        return f"""{death_info}。

{get_json_prompt_instructions(WitchActionModelCN)}

注意：
- 解药只能使用一次，请谨慎选择
- 毒药也只能使用一次
"""
    
    @staticmethod
    def get_hunter_prompt(players: List[Any]) -> str:
        """获取猎人开枪提示"""
        return f"""你即将出局，请选择是否开枪带走一名玩家。

{get_json_prompt_instructions(get_hunter_model_cn(players))}

可选目标：{', '.join([p.name for p in players])}
"""
    
    @staticmethod
    def get_vote_prompt(players: List[Any]) -> str:
        """获取投票提示"""
        model = get_vote_model_cn(players)
        return f"""请投票选择要淘汰的玩家。

{get_json_prompt_instructions(model)}

可选目标：{', '.join([p.name for p in players])}
"""