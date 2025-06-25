from model import Model
from typing import List


class TableCleanup:
    def __init__(self, table_data: List[List[str]], context: str, model: Model):
        self.table_data = table_data
        self.context = context
        self.model = model
        self.instruction = self._build_prompt()

    def _format_table_for_prompt(self):
        """Convert list of lists to readable table format"""
        formatted_rows = []
        for row in self.table_data:
            # Join cells with | separator
            formatted_row = " | ".join(str(cell) for cell in row)
            formatted_rows.append(formatted_row)

        return "\n".join(formatted_rows)

    def _build_prompt(self):
        formatted_table = self._format_table_for_prompt()

        return f"""
## Your task:
You are given a table extracted from a PDF document with the context of the document. Your task is to analyze and describe the table content into a representative text for vector database storage by following these steps:
1. Table understanding
2. Content extraction and interpretation
3. Summarized description generation

## Context:
{self.context}

## Table data:
{formatted_table}

## remarks:
- Table may contain english or french text, or both
- Describe the key information and patterns by considering the context and the column headers
- Output should be plain text

## The table summary:
        """

    def process(self):
        messages = [
            {
                "role": "system",
                "content": "You are a professional table analyzer and content extractor specialized in academic and educational materials.",
            },
            {"role": "user", "content": self.instruction},
        ]
        return self.model.generate(messages)
