import os
from typing import Dict, List, Optional

from ..judge.agent_as_a_judge.llm.provider import LLM
from .token_counter import log_token_usage as log_agent_token_usage


class TextModifierAgent:
    """文本修改 Agent：
    - 系统提示词可与生成 Agent 不同
    - 根据 Judge 奖励信号（建议/未满足点）对现有文本进行有向改写
    - 仅输出改写后的文本
    """

    def __init__(self, system_prompt: Optional[str] = None, model_name: Optional[str] = None):
        default_sys = (
            "You are an expert writing editor. Revise the given text to better satisfy the rubric and suggestions. "
            "Preserve factual correctness, avoid hallucinations, keep consistent tone and structure. "
            "Return ONLY the revised text."
        )
        self.system_prompt = system_prompt or default_sys
        # 允许自由指定 TEXT Agent 的模型；未指定则回退到环境变量
        self.llm = LLM(model=model_name)

    def modify(
        self,
        current_text: str,
        task_input: str,
        rubric: Dict,
        judge_suggestions: List[str],
        temperature: float = 1,
        use_judge_suggestions: bool = True,
        section_key: str=None
    ) -> Dict:
        
        if use_judge_suggestions and (judge_suggestions or []):
            sug = "\n".join(f"- {s}" for s in (judge_suggestions or []))
            user = (
                "Task input (for context):\n" + task_input +
                "\n\nRubric (JSON):\n" + str(rubric) +
                "\n\nCurrent text:\n" + current_text +
                "\n\nRevise the text. Strict rules:\n- Address the suggestions below.\n- Improve to reach higher rubric level(s).\n- Do not add external unverifiable facts.\n- Return ONLY revised text.\n\nSuggestions:\n" + sug
            )
        else:
            # 消融设定：不参考 Judge 的建议，直接进行多样化变异，专注于 rubric 要求
            user = (
                "Task input (for context):\n" + task_input +
                "\n\nRubric (JSON):\n" + str(rubric) +
                "\n\nCurrent text:\n" + current_text +
                "\n\nRevise the text to better satisfy the rubric without using judge suggestions.\n"
                "Make a direct mutation (e.g., refine structure, reasoning depth, style, or specificity) while keeping factual correctness.\n"
                "Return ONLY the revised text."
            )
        # 实验章节：在发送消息前追加“图环境保留”强约束，防止 LLM 删除或改写 figure 代码块
        try:
            if (task_input or "").lower().find("target section: experiment") != -1:
                user = user + (
                    "\n\nEXPERIMENT SECTION — FIGURE PRESERVATION RULES:\n"
                    "- Preserve ALL LaTeX figure environments EXACTLY as in the current text.\n"
                    "- Do NOT remove or modify any \\begin{figure}...\\end{figure} blocks.\n"
                    "- Do NOT change any \\includegraphics path/options, \\caption, or \\label.\n"
                    "- Do NOT convert figures into prose; keep code blocks intact.\n"
                    "- You may edit surrounding prose, but NEVER edit inside figure environments.\n"
                )
        except Exception:
            pass

        # 介绍或相关工作章节不允许删改正文的文献引用
        if section_key == "related_work" or section_key == "introduction":

            if section_key == "introduction":
                messages = [{"role": "system", "content": self.system_prompt+"\nIt is necessary to introduce with a natural transition sentence at the end, and then list the summary of contributions (list/items of contributions), so as to clearly and structurally summarize the main contributions.\n"+"\n# Be sure to retain all references and citations, and do not remove any of the cited literature. For example, do not delete or modify any citation commands such as \cite{key1, key2}, \citep{key}, \citeauthor{key}, etc., or their corresponding reference entries.\n"}, 
                            {"role": "user", "content": "**Be sure to retain all references and citations, and do not remove any of the cited literature. For example, do not delete or modify any citation commands such as \cite{key1, key2}, \citep{key}, \citeauthor{key}, etc., or their corresponding reference entries.**\n"+user}]
            else:
                messages = [{"role": "system", "content": self.system_prompt+"\n# Be sure to retain all references and citations, and do not remove any of the cited literature. For example, do not delete or modify any citation commands such as \cite{key1, key2}, \citep{key}, \citeauthor{key}, etc., or their corresponding reference entries.\n"}, 
                            {"role": "user", "content": "**Be sure to retain all references and citations, and do not remove any of the cited literature. For example, do not delete or modify any citation commands such as \cite{key1, key2}, \citep{key}, \citeauthor{key}, etc., or their corresponding reference entries.**\n"+user}]
            
        else:
            messages = [{"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user}]

        resp, cost, _acc = self.llm.do_completion(messages=messages, temperature=temperature)
        text = resp.choices[0].message["content"]
        usage = getattr(resp, "usage", None)
        result = {
            "text": text,
            "llm_stats": {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) if usage else 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) if usage else 0),
                "cost": float(cost or 0.0),
            },
        }

        # 分开记录 TEXT Agent 的 token 与成本日志（写入 mas/logs）
        try:
            log_agent_token_usage(
                model=self.llm.model_name,
                prompt_tokens=result["llm_stats"]["input_tokens"],
                completion_tokens=result["llm_stats"]["output_tokens"],
                total_tokens=result["llm_stats"]["input_tokens"] + result["llm_stats"]["output_tokens"],
                cost=result["llm_stats"]["cost"],
            )
        except Exception:
            pass

        return result
