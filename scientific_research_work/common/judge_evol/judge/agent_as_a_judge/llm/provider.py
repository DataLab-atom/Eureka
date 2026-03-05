import warnings
import time
from functools import partial
import os

from llm.env import get_env, load_env

load_env()
my_api_key = get_env("OPENAI_API_KEY") or get_env("API_KEY") or ""
my_base_url = (
    get_env("OPENAI_BASE_URL")
    or get_env("OPENAI_API_BASE")
    or get_env("LITELLM_BASEURL")
    or get_env("LITELLM_BASE_URL")
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import litellm

from litellm import completion as litellm_completion
from litellm import completion_cost as litellm_completion_cost
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from ...token_counter import log_token_usage

# Optional verbose logging if explicitly set (never enable by default to avoid leaking credentials)
if os.getenv("LITELLM_VERBOSE", "").lower() in ("1", "true", "yes", "on"):
    os.environ["LITELLM_LOG"] = "DEBUG"
    try:
        litellm.set_verbose = True
    except Exception:
        pass

__all__ = ["LLM"]

message_separator = "\n\n----------\n\n"


class LLM:
    def __init__(
        self,
        model=None,
        api_key=my_api_key,
        base_url=my_base_url,
        api_version=None,
        num_retries=3,
        retry_min_wait=1,
        retry_max_wait=10,
        llm_timeout=900,
        llm_temperature=1,
        llm_top_p=0.9,
        custom_llm_provider=None,
        max_input_tokens=16384,
        max_output_tokens=8192,
        cost=None,
    ):

        from .cost import Cost

        self.cost = Cost()
        # Resolve configuration from env if not provided
        self.model_name = model

        # Allow overriding base URL via common env vars
        if base_url is None:
            base_url = (
                os.getenv("LITELLM_BASEURL")
                or os.getenv("LITELLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or os.getenv("OPENAI_API_BASE_URL")
            )
        self.base_url = base_url
        self.api_key = api_key

        # Azure/OpenAI API version, if needed
        self.api_version = api_version or os.getenv("OPENAI_API_VERSION")
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.llm_timeout = llm_timeout
        self.llm_temperature = llm_temperature
        self.llm_top_p = llm_top_p
        self.num_retries = num_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.custom_llm_provider = custom_llm_provider

        self.model_info = None
        try:
            self.model_info = litellm.get_model_info(self.model_name)
        except Exception:
            print(f"Could not get model info for {self.model_name}")

        if self.max_input_tokens is None and self.model_info:
            self.max_input_tokens = self.model_info.get("max_input_tokens", 4096)
        if self.max_output_tokens is None and self.model_info:
            self.max_output_tokens = self.model_info.get("max_output_tokens", 1024)

        self._initialize_completion_function()

    def _initialize_completion_function(self):
        # 对 gpt-5 系列推理模型，不传 temperature/top_p
        base_kwargs = {
            "model": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "custom_llm_provider": self.custom_llm_provider,
            "max_tokens": self.max_output_tokens,
            "timeout": self.llm_timeout,
        }
        if not self._is_gpt5():
            base_kwargs["temperature"] = self.llm_temperature
            base_kwargs["top_p"] = self.llm_top_p
        completion_func = partial(litellm_completion, **base_kwargs)

        def attempt_on_error(retry_state):
            print(f"Could not get model info for {self.model_name}")
            return True

        @retry(
            reraise=True,
            stop=stop_after_attempt(self.num_retries),
            wait=wait_random_exponential(
                min=self.retry_min_wait, max=self.retry_max_wait
            ),
            retry=retry_if_exception_type(
                (RateLimitError, APIConnectionError, ServiceUnavailableError)
            ),
            after=attempt_on_error,
        )
        def wrapper(*args, **kwargs):

            resp = completion_func(*args, **kwargs)
            message_back = resp["choices"][0]["message"]["content"]
            # logger.debug(message_back)
            return resp, message_back

        self._completion = wrapper

    @property
    def completion(self):
        return self._completion

    def _llm_inference(self, messages: list) -> dict:
        """Perform LLM inference using deterministic temperature 0.0."""
        return self._llm_inference_with_temperature(messages, temperature=0.0)

    def _llm_inference_with_temperature(self, messages: list, temperature: float) -> dict:
        """Perform LLM inference using the provided messages and temperature."""
        start_time = time.time()
        call_kwargs = {"messages": messages}
        # gpt-5 系列不传 temperature
        if not self._is_gpt5():
            call_kwargs["temperature"] = temperature
        response, cost, accumulated_cost = self.do_completion(**call_kwargs)
        inference_time = time.time() - start_time

        llm_response = response.choices[0].message["content"]
        input_token, output_token = (
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

        log_token_usage(
            model=self.model_name,
            prompt_tokens=input_token,
            completion_tokens=output_token,
            total_tokens=input_token + output_token,
            cost=cost
        )

        return {
            "llm_response": llm_response,
            "input_tokens": input_token,
            "output_tokens": output_token,
            "cost": cost,
            "accumulated_cost": accumulated_cost,
            "inference_time": inference_time,
        }

    def _is_gpt5(self) -> bool:
        try:
            return str(self.model_name or "").lower().startswith("gpt-5")
        except Exception:
            return False

    def do_completion(self, *args, **kwargs):
        resp, msg = self._completion(*args, **kwargs)
        cur_cost, accumulated_cost = self.post_completion(resp)
        return resp, cur_cost, accumulated_cost

    def post_completion(self, response: str):
        try:
            cur_cost = self.completion_cost(response)
        except Exception:
            cur_cost = 0

        return cur_cost, self.cost.accumulated_cost  # , cost_msg

    def get_token_count(self, messages):
        return litellm.token_counter(model=self.model_name, messages=messages)

    def is_local(self):
        if self.base_url:
            return any(
                substring in self.base_url
                for substring in ["localhost", "127.0.0.1", "0.0.0.0"]
            )
        if self.model_name and self.model_name.startswith("ollama"):
            return True
        return False

    def completion_cost(self, response):
        if not self.is_local():
            try:
                cost = litellm_completion_cost(completion_response=response)
                if self.cost:
                    self.cost.add_cost(cost)
                return cost
            except Exception:
                print("Cost calculation not supported for this model.")
        return 0.0

    def __str__(self):
        return f"LLM(model={self.model_name}, base_url={self.base_url})"

    def __repr__(self):
        return str(self)

    def do_multimodal_completion(self, text, image_path):
        messages = self.prepare_messages(text, image_path=image_path)
        response, cur_cost, accumulated_cost = self.do_completion(messages=messages)
        return response, cur_cost, accumulated_cost

    @staticmethod
    def encode_image(image_path):
        import base64

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def prepare_messages(self, text, image_path=None):
        messages = [{"role": "user", "content": text}]
        if image_path:
            base64_image = self.encode_image(image_path)
            messages[0]["content"] = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + base64_image},
                },
            ]
        return messages


if __name__ == "__main__":
    load_env()

    model_name = "gpt-4o-2024-08-06"
    api_key = get_env("OPENAI_API_KEY")
    base_url = get_env("OPENAI_BASE_URL") or "https://api.openai.com/v1"

    llm_instance = LLM(model=model_name, api_key=api_key, base_url=base_url)

    image_path = "/Users/zhugem/Desktop/DevAI/studio/workspace/sample/results/prediction_interactive.png"

    for i in range(1):

        multimodal_response = llm_instance.do_multimodal_completion(
            "What’s in this image?", image_path
        )
        print(multimodal_response)
