from __future__ import annotations

from llm.types import LLMPreset


class LLM_PRESETS:
    CHART_DESCRIPTION = LLMPreset(name="chart_description")
    CHECK_REFERENCE_REPEAT = LLMPreset(name="check_reference_repeat", json_mode=True)
    PLOTCODE_DESCRIBE = LLMPreset(name="plotcode_describe", json_mode=True)
