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
        context: str = None,
        model: Model = None,
        output_schema: str = OUTPUT_SCHEMA,
    ):
        if (
            isinstance(image_data, dict)
            and "base64" in image_data.keys()
            and "ext" in image_data.keys()
        ):
            if not isinstance(image_data["base64"], str):
                raise ValueError("image 'base64' must be a base64 string representation.")
            self.img = image_data["base64"]
            self.ext = image_data["ext"]
        else:
            raise ValueError("Image data must be a dictionary with 'base64' and 'ext' keys.")

        self.output_schema = output_schema
        self.model = model
        self.instruction = PromptTemplate(
            template=IMAGE_CLEANUP_PROMPT, 
            image_extension=self.ext,
            context=context
        ).prompt
        self.system_instruction = IMAGE_CLEANUP_SYSTEM_PROMPT

    def process(self):
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
