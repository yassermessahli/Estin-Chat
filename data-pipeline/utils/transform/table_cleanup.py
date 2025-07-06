from typing import List
from .model import Model
from .prompts import PromptTemplate, TABLE_CLEANUP_PROMPT, TABLE_CLEANUP_SYSTEM_PROMPT, OUTPUT_SCHEMA


class TableCleanup:
    def __init__(self, table_data: List[List[str]] = None, model: Model = None, output_schema: str = OUTPUT_SCHEMA):
        self.table_data = table_data
        self.model = model
        self.output_schema = output_schema
        
        self.instruction = PromptTemplate(template=TABLE_CLEANUP_PROMPT, table_data=self._format_table_for_prompt()).prompt
        self.system_instruction = TABLE_CLEANUP_SYSTEM_PROMPT
        
    def _format_table_for_prompt(self):
        """Convert list of lists to readable table format"""
        if self.table_data:
            formatted_rows = []
            for row in self.table_data:
                # Join cells with | separator
                formatted_row = " | ".join(str(cell).replace("\n", "/") for cell in row)
                formatted_rows.append(formatted_row)
            return "\n".join(formatted_rows)
        return "No table data"

    def process(self):
        messages = [
            {
                "role": "system",
                "content": self.system_instruction,
            },
            {"role": "user", "content": self.instruction},
        ]
        return self.model.generate(messages).message.content
    
if __name__ == "__main__":
    # Example usage
    table_data =  [["",
    "𝒄𝒐𝒔()",
    "𝒔𝒊𝒏()",
    "𝒂 = 𝒁𝒄𝒐𝒔()",
    "𝒃 = 𝒁𝒔𝒊𝒏()",
    "𝒛 = 𝒁𝒆𝒋",
    "𝒆𝒋"
    ],
    ["0", "1", "0", "𝑍", "0", "𝑍", "1"],
    ["𝜋\n2", "0", "+1", "0", "𝑍", "𝑗𝑍", "𝑗"],
    ["𝜋\n−\n2", "0", "-1", "0", "−𝑍", "−𝑗𝑍", "−𝑗"]]
    
    res = TableCleanup(table_data=table_data).instruction
    print(res)  # Output the cleaned table content
