import tiktoken
from transformers import AutoTokenizer


class NUM_TOKENS_FROM_STRING:
    def __init__(self) -> None:
        self.tokenizer_dict = {}
        self.tokenizer_dict["gpt-4.1-mini"] = tiktoken.encoding_for_model("gpt-4.1-mini")
        self.tokenizer_dict["gpt-4.1"] = tiktoken.encoding_for_model("gpt-4.1")
        self.tokenizer_dict["text-embedding-3-large"] = tiktoken.encoding_for_model("text-embedding-3-large")
        self.tokenizer_dict["gpt-5-mini"] = tiktoken.encoding_for_model("gpt-5-mini")
        self.tokenizer_dict["gpt-4.1-nano"] = tiktoken.encoding_for_model("gpt-4.1")
        self.tokenizer_dict["gpt-5"] = tiktoken.encoding_for_model("gpt-4.1-mini")
        self.tokenizer_dict["BAAI/bge-m3"] = AutoTokenizer.from_pretrained("BAAI/bge-m3", timeout=30)

    def get_string_num_tokens(self, string: str, model_name: str) -> int:
        return len(self.tokenizer_dict[model_name].encode(string))


num_tokens_from_string = NUM_TOKENS_FROM_STRING()
