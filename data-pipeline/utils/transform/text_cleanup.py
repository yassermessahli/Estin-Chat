from .model import Model
from .prompts import (
    PromptTemplate,
    TEXT_CLEANUP_PROMPT,
    TEXT_CLEANUP_SYSTEM_PROMPT,
    OUTPUT_SCHEMA,
)


class TextCleanup:
    def __init__(
        self, text: str = None, model: Model = None, output_schema: str = OUTPUT_SCHEMA, context: str = None
    ):
        self.text = text
        self.model = model
        self.output_schema = output_schema
        self.context = context

        self.instruction = PromptTemplate(
            template=TEXT_CLEANUP_PROMPT, text=self.text, context=self.context
        ).prompt
        self.system_instruction = TEXT_CLEANUP_SYSTEM_PROMPT

    def process(self):
        if self.text:
            messages = [
                {
                    "role": "system",
                    "content": self.system_instruction,
                },
                {"role": "user", "content": self.instruction},
            ]
            return self.model.generate(messages).messages[0].content
        raise ValueError("No text provided for cleanup.")
