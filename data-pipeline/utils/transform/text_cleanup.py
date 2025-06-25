from model import Model


class TextCleanup:
    def __init__(self, text: str, model: Model):
        self.text = text
        self.model = model
        self.instruction = self._build_prompt()

    def _build_prompt(self):
        return f"""
## Your task:
You are given the raw text extracted from a PDF document that contains some noises and formatting issues. Your task is to clean it and format it for vector database populating by following the next steps:
1. text understanding
1. cleaning
2. summarizing
3. standardizing to paragraphs format with two endlines between each paragraph

## remarks:
- try to make the output a plain text.
- consider formatting the math expressions.
- content may be in english or french, output should be exclusively in english.
- try to keep the original expression and meaning of the text. 
- put the summary directly, no introductions from you.
- it may contains headers and footers, you should ignore them and concider only the main content.

## The text:
{self.text}

## The output:
        """

    def process(self):
        messages = [
            {
                "role": "system",
                "content": "You are a professional text cleaner and summarizer for both english and french raw texts.",
            },
            {"role": "user", "content": self.instruction},
        ]
        return self.model.generate(messages)
