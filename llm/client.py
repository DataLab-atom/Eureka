from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from openai import OpenAI

from llm.env import get_env
from llm.logging import input_logger, output_logger
from llm.tokens import num_tokens_from_string
from llm.types import LLMPreset


def get_client(model_name: str | None = None) -> OpenAI:
    api_key = get_env("OPENAI_API_KEY", "") or ""
    base_url = get_env("OPENAI_BASE_URL")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


class LLMClient:
    def __init__(
        self,
        default_model: Optional[str] = None,
        default_temperature: Optional[float] = None,
        max_retries: int = 3,
    ) -> None:
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.max_retries = max_retries

    def call(
        self,
        messages: List[Dict[str, Any]],
        preset: Optional[LLMPreset] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        json_mode: Optional[bool] = None,
    ) -> Any:
        resolved_model = model or (preset.model if preset else None) or self.default_model
        if not resolved_model:
            raise ValueError("model is required for LLM call")

        resolved_temperature = (
            temperature
            if temperature is not None
            else (preset.temperature if preset else None)
            if preset is not None
            else self.default_temperature
        )

        resolved_schema = response_schema or (preset.response_schema if preset else None)
        resolved_json_mode = (
            json_mode
            if json_mode is not None
            else (preset.json_mode if preset else False)
            if preset is not None
            else False
        )

        client = get_client(resolved_model)
        self._log_input_tokens(messages, resolved_model)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if resolved_schema is not None:
                    completion = client.beta.chat.completions.parse(
                        messages=messages,
                        model=resolved_model,
                        response_format=resolved_schema,
                        temperature=resolved_temperature,
                    )
                    parsed = completion.choices[0].message.parsed
                    result = dict(parsed)
                    self._log_output_tokens(json.dumps(result), resolved_model)
                    return result

                if resolved_json_mode:
                    completion = client.chat.completions.create(
                        messages=messages,
                        model=resolved_model,
                        response_format={"type": "json_object"},
                        temperature=resolved_temperature,
                    )
                    content = completion.choices[0].message.content
                    self._log_output_tokens(content, resolved_model)
                    return self._parse_json(content)

                completion = client.chat.completions.create(
                    messages=messages,
                    model=resolved_model,
                    temperature=resolved_temperature,
                )
                content = completion.choices[0].message.content
                self._log_output_tokens(content, resolved_model)
                return content
            except Exception as exc:
                last_error = exc
                logging.error("LLM call failed on attempt %s: %s", attempt, exc)

        raise RuntimeError("LLM call failed after retries") from last_error

    def _parse_json(self, content: str) -> Dict[str, Any]:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            markdown_match = re.search(r"```json\s*([\s\S]*?)\s*```", content, re.IGNORECASE)
            if markdown_match:
                return json.loads(markdown_match.group(1).strip())
            json_match = re.search(r"^\s*\{.*\}\s*$", content, re.DOTALL)
            if json_match:
                return json.loads(content.strip())
            raise

    def _log_input_tokens(self, messages: List[Dict[str, Any]], model: str) -> None:
        try:
            input_logger.info(num_tokens_from_string.get_string_num_tokens(json.dumps(messages), model))
        except Exception:
            return

    def _log_output_tokens(self, content: str, model: str) -> None:
        try:
            output_logger.info(num_tokens_from_string.get_string_num_tokens(content, model))
        except Exception:
            return
