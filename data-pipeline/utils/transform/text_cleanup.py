from .model import Model
from .prompts import PromptTemplate, TEXT_CLEANUP_PROMPT, TEXT_CLEANUP_SYSTEM_PROMPT, OUTPUT_SCHEMA

class TextCleanup:
    def __init__(self, text: str = "", model: Model = None, output_schema: str = OUTPUT_SCHEMA):
        self.text = text if text is not None else "No text provided."
        self.model = model
        self.output_schema = output_schema

        self.instruction = PromptTemplate(template=TEXT_CLEANUP_PROMPT, text=self.text).prompt
        self.system_instruction = TEXT_CLEANUP_SYSTEM_PROMPT
        
    def process(self):
        messages = [
            {
                "role": "system",
                "content": self.system_instruction,
            },
            {"role": "user", "content": self.instruction},
        ]
        return self.model.generate(messages).messages[0].content
