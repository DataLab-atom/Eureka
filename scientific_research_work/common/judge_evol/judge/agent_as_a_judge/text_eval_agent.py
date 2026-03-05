import json
import os
from typing import Any, Dict, List, Optional

from .llm.provider import LLM


class TextEvalAgent:
    """
    纯 LLM 文本评估 Agent（无外部工具），按给定 rubric 对文本打分并给出修改建议。

    输入：
      - text: 待评估文本
      - task_input: 原始任务/输入文本（为评估提供上下文，不参与评分输出）
      - rubric: dict，期望包含以下键（与 BIGGen 示例兼容）：
          {
            "criteria": "...",
            "score1_description": "...",
            "score2_description": "...",
            "score3_description": "...",
            "score4_description": "...",
            "score5_description": "..."
          }

    输出（dict）：
      {
        "scores": [int],            # 1-5 之间的整数数组；若仅一条标准则仅返回一个分数
        "mean_score": float,        # 平均分（四舍五入到 2 位）
        "reasons": [str],           # 每个分数的评语/依据
        "suggestions": [str],       # 修改建议（奖励信号）
        "raw": str,                 # 原始 LLM 输出（便于排错）
        "llm_stats": {              # token / cost / time 统计
          "input_tokens": int,
          "output_tokens": int,
          "cost": float,
          "inference_time": float
        }
      }
    """

    def __init__(self, model_name: Optional[str] = None):
        # 优先级：显式参数 > 环境变量 JUDGE_LLM > 环境变量 DEFAULT_LLM
        model = model_name or os.getenv("JUDGE_LLM") or os.getenv("DEFAULT_LLM")
        self.llm = LLM(model=model)

    def _build_messages(self, text: str, rubric: Dict[str, Any], task_input: Optional[str] = None) -> List[Dict[str, str]]:
        rubric_json = json.dumps(rubric, ensure_ascii=False)
        system = (
            "You are a strict, consistent writing evaluator. "
            "Score the response using the provided rubric only. "
            "Return STRICT JSON with keys: scores (int[]), mean_score (float), reasons (str[]), suggestions (str[]). "
            "Scores must be integers in [1,5]. Be concrete and actionable in suggestions. No extra text."
        )
        task_part = "\n\nOriginal task/input:\n" + (task_input or "") if task_input is not None else ""
        user = (
            "Rubric JSON:\n" + rubric_json +
            task_part +
            "\n\nResponse to evaluate (plain text):\n" + text +
            "\n\nRules:\n- If rubric provides a single criterion with 1..5 descriptions, output a single score in 'scores'.\n"
            "- Base your reasons on mismatches between response and the rubric's level descriptions.\n"
            "- Suggestions should directly guide revision to reach higher score levels.\n"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _safe_parse_json(text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        # 尝试提取首个 JSON 对象
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def evaluate(self, text: str, rubric: Dict[str, Any], task_input: Optional[str] = None) -> Dict[str, Any]:
        messages = self._build_messages(text, rubric, task_input)
        res = self.llm._llm_inference(messages)
        raw = (res.get("llm_response") or "").strip()
        data = self._safe_parse_json(raw)

        print(data)

        scores = data.get("scores")
        if not isinstance(scores, list) or not scores:
            # 回退：若无法解析则返回一个 0 分并附带建议
            scores = [0]

        # 归一化分数
        norm_scores = []
        for s in scores:
            try:
                si = int(s)
            except Exception:
                si = 0
            si = max(0, min(5, si))
            norm_scores.append(si)

        reasons = data.get("reasons")
        if not isinstance(reasons, list):
            reasons = [str(reasons)] if reasons is not None else []

        suggestions = data.get("suggestions")
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)] if suggestions is not None else []

        mean_score = sum(norm_scores) / len(norm_scores) if norm_scores else 0.0
        mean_score = round(float(mean_score), 2)

        return {
            "scores": norm_scores,
            "mean_score": mean_score,
            "reasons": reasons,
            "suggestions": suggestions,
            "raw": raw,
            "llm_stats": {
                "input_tokens": int(res.get("input_tokens", 0)),
                "output_tokens": int(res.get("output_tokens", 0)),
                "cost": float(res.get("cost", 0.0)),
                "inference_time": float(res.get("inference_time", 0.0)),
            },
        }
    


#  "You are a strict evaluator. Given a response and a checklist of decomposed questions, "
# "determine for EACH question whether the response satisfies it. "
# "Return STRICT JSON: {\"booleans\":[true|false], \"reasons\":[string], \"suggestions\":[string]}. "
# "The length of arrays must equal the number of questions, no extra text."


class TextJudgeAgent:
    """
    针对 text 的评估 Agent：
    """

    def __init__(self, model_name: Optional[str] = None):
        # 优先级：显式参数 > 环境变量 JUDGE_LLM > 环境变量 DEFAULT_LLM
        model = model_name or os.getenv("JUDGE_LLM") or os.getenv("DEFAULT_LLM")
        self.llm = LLM(model=model, llm_temperature=0)

    def _build_messages(self, text: str, questions: list, explain_reasons: bool=False) -> list: # task_input: Optional[str] = None
        system = (
            "You are a strict evaluator. Given a response and a checklist of decomposed questions, "+
            "determine for EACH question whether the response satisfies it. "+
            "Return STRICT JSON: {\"booleans\":[true|false]"+
            (', \"reasons\":[string]}.' if explain_reasons else '}.')+
            "The length of arrays must equal the number of questions, no extra text."
        )
        user = (
            "\n# Rules:\n- booleans[i] indicates whether question[i] is satisfied.\n" + ('\n- reasons[i] briefly justify the decision.' if explain_reasons else '') + "\n# Decomposed questions as a JSON array:\n" + json.dumps(questions, ensure_ascii=False) +
            # ("\n\nOriginal task/input:\n" + task_input if task_input else "") +
            "\n\n# Response to evaluate:\n" + text
            # "\n\nRules:\n- booleans[i] indicates whether question[i] is satisfied.\n- reasons[i] briefly justify the decision.\n- suggestions[i] give concrete guidance to satisfy question[i] if false; can be empty if true.\n"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def _safe_parse_json(text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def evaluate(self, text: str, questions: list, explain_reasons: bool=False) -> Dict[str, Any]:  # task_input: Optional[str] = None
        messages = self._build_messages(text, questions, explain_reasons)  # task_input
        res = self.llm._llm_inference(messages)
        raw = (res.get("llm_response") or "").strip()
        data = self._safe_parse_json(raw)

        booleans = data.get("booleans")
        if not isinstance(booleans, list) or not booleans:
            # 回退：全部判为 False
            booleans = [False for _ in range(len(questions))]

        # 归一化为 bool 列表
        norm_bools: List[bool] = []
        for b in booleans:
            if isinstance(b, bool):
                norm_bools.append(b)
            elif isinstance(b, str):
                norm_bools.append(b.strip().lower() in ("true", "yes", "y", "1"))
            else:
                try:
                    norm_bools.append(bool(b))
                except Exception:
                    norm_bools.append(False)

        # 长度对齐
        if len(norm_bools) != len(questions):
            # 截断或填充为 False
            if len(norm_bools) > len(questions):
                norm_bools = norm_bools[: len(questions)]
            else:
                norm_bools += [False] * (len(questions) - len(norm_bools))

        if explain_reasons:
            reasons = data.get("reasons")
            if not isinstance(reasons, list):
                reasons = [str(reasons)] if reasons is not None else []
            if len(reasons) != len(questions):
                # 对齐长度（过短则填空）
                reasons = (reasons + [""] * len(questions))[: len(questions)]

        # suggestions = data.get("suggestions")
        # if not isinstance(suggestions, list):
        #     suggestions = [str(suggestions)] if suggestions is not None else []
        # if len(suggestions) != len(questions):
        #     # 对于未通过项，若无建议则给出基于题目的默认建议
        #     tmp = (suggestions + [None] * len(questions))[: len(questions)]
        #     suggestions = [
        #         (s if (s is not None and str(s).strip()) else f"Ensure: {q}") if not norm_bools[i] else ""
        #         for i, (s, q) in enumerate(zip(tmp, questions))
        #     ]

        passed = sum(1 for b in norm_bools if b)
        total = max(1, len(norm_bools))
        pass_rate = round(passed / total, 4)

        return {
            "scores": norm_bools,  # 布尔数组
            "mean_score": pass_rate,  # 通过率（通过数/总数）
            **({"reasons": reasons} if explain_reasons else {}),
            # "suggestions": suggestions,
            "raw": raw,
            "llm_stats": {
                "input_tokens": int(res.get("input_tokens", 0)),
                "output_tokens": int(res.get("output_tokens", 0)),
                "cost": float(res.get("cost", 0.0)),
                "inference_time": float(res.get("inference_time", 0.0)),
            },
        }


