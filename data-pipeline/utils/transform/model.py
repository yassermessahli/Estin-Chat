from ollama import Client, ChatResponse
from dataclasses import dataclass


@dataclass
class ModelParams:
    host: str | None = "http://localhost:11434"
    model: str | None = None
    think: bool | None = True
    temperature: float | None = 0.5
    num_predict: int | None = None
    num_ctx: int | None = None
    sys_message: str | None = None


class Model:
    def __init__(self, params: ModelParams):
        self.params = params
        self.base = Client(host=self.params.host)

    def _get_valid_params(self):
        return {
            k: v
            for k, v in {
                "temperature": self.params.temperature,
                "num_predict": self.params.num_predict,
                "num_ctx": self.params.num_ctx,
            }.items()
            if v is not None
        }

    def generate(self, prompts) -> ChatResponse:
        return self.base.chat(
            model=self.params.model,
            messages=prompts,
            think=self.params.think,
            options=self._get_valid_params(),
        )
