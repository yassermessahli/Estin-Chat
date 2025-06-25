from model import Model
import base64
from pathlib import Path


class ImageCleanup:
    def __init__(self, image_data, context: str, model: Model):
        self.image_data = image_data
        self.context = context
        self.model = model
        self.instruction = self._build_prompt()

    def _build_prompt(self):
        return f"""
## Your task:
give a description for the given ({self.image_data["ext"]}) image using the given context by following these steps:
1. Image understanding
2. Content extraction (text, diagrams, charts, formulas, etc.)
3. summarized description generation

## Context:
{self.context}

## remarks:
- Image may contain english or french text, output should exclusively be in english.
- Be accurate
- outputs should be a very short paragraph summarizing the image content in plain text.

## the image summary here:
The image represents ...
        """

    def process(self):
        # Convert binary data to base64 for the vision model
        # image_b64 = base64.b64encode(self.image_data).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": "You are an accurate image analyzer and content extractor from academic and educational materials.",
            },
            {
                "role": "user",
                "content": self.instruction,
                "images": [self.image_data["base64"]],  # Add the base64 image data here
            },
        ]
        return self.model.generate(messages)
