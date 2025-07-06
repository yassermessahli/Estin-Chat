from .model import Model
from .prompts import (
    PromptTemplate,
    IMAGE_CLEANUP_PROMPT,
    IMAGE_CLEANUP_SYSTEM_PROMPT,
    OUTPUT_SCHEMA,
)


class ImageCleanup:
    def __init__(
        self,
        image_data: dict = None,
        model: Model = None,
        output_schema: str = OUTPUT_SCHEMA,
    ):
        self.img = image_data["base64"] if image_data is not None else ""
        self.ext = image_data["ext"] if image_data is not None else ""
        self.model = model

        self.instruction = PromptTemplate(
            template=IMAGE_CLEANUP_PROMPT, image_extension=self.ext
        ).prompt
        self.system_instruction = IMAGE_CLEANUP_SYSTEM_PROMPT

    def process(self):
        # IF NEEDED: Convert binary data to base64 for the vision model
        # image_b64 = base64.b64encode(self.image_data).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": self.system_instruction,
            },
            {
                "role": "user",
                "content": self.instruction,
                "images": [self.img],
            },
        ]
        return self.model.generate(messages).message.content
