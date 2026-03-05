import json
import os
from typing import Any, Dict, List, Optional

from .llm.provider import LLM


class MultimodalJudgeAgent:
    """
    针对 Multimodal 的评估 Agent：
    """

    def __init__(self, model_name: Optional[str] = None):
        # 优先级：显式参数 > 环境变量 JUDGE_LLM > 环境变量 DEFAULT_LLM
        model = model_name or os.getenv("JUDGE_LLM") or os.getenv("DEFAULT_LLM")
        self.llm = LLM(model=model, llm_temperature=0)

    def _build_messages(self, text: str, questions: list, explain_reasons: bool=False) -> list:  # task_input: Optional[str] = None, 
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
        return [{"role": "system", "content": system}, {"role": "user", "content": [{'type': 'text', 'text': user}]}]

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

    def evaluate(self, text: str, images: list, questions: list, explain_reasons: bool=False) -> Dict[str, Any]:
        messages = self._build_messages(text, questions, explain_reasons)

        # 处理图像内容
        for image in images:
            messages[1]['content'].append( {
                                    "type": "image_url",
                                    "image_url": {
                                    "url": f"data:image/jpeg;base64,{image['reference content']['image_base64']}"
                                    }
                                }
                            )
        
            messages[1]['content'].append({'type': 'text', 
                                           'text': f"The relevant information for this reference image is:\n{json.dumps({'Image path': image['reference content']['image_path'], 'Bib citation': image['image_references']}, ensure_ascii=False)}"
                                           })
            
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
            "raw": raw,
            "llm_stats": {
                "input_tokens": int(res.get("input_tokens", 0)),
                "output_tokens": int(res.get("output_tokens", 0)),
                "cost": float(res.get("cost", 0.0)),
                "inference_time": float(res.get("inference_time", 0.0)),
            },
        }



