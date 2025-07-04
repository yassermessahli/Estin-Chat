from .model import Model


class TextCleanup:
    def __init__(self, text: str, model: Model):
        self.text = text
        self.model = model
        self.instruction = self._build_prompt()

    def _build_prompt(self):
        return f"""
## Your task:
You are given the raw text extracted from a PDF document that contains some noises and formatting issues. Your task is to clean it and format it for vector storing by following the next steps:
1. text understanding
1. cleaning
2. summarizing
3. standardizing to paragraphs format

## remarks:
- consider reformatting the math expressions.
- content may be in english or french, output should be exclusively in english.
- try to keep the original expression and meaning of the text. 
- put the summary directly, no introductions from you.
- it may contains headers and footers and authors names, you should ignore them and consider only the main content.
- the output should be in paragraphs separated by an empty line. each paragraph composed of setences separated by dots.

## The text:
{self.text}

## Your output:
        """

    def process(self):
        messages = [
            {
                "role": "system",
                "content": "You are a professional text cleaner and summarizer for both english and french raw texts.",
            },
            {"role": "user", "content": self.instruction},
        ]
        return self.model.generate(messages).messages[0].content
