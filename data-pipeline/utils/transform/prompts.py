# ./prompts.py

TEXT_CLEANUP_PROMPT = """
## Task  
Clean and structure raw text from a PDF for vector storage:

1. Understand content.  
2. Remove noise (headers, footers, author names).  
3. Reformat math expressions.  
4. Summarize and split into semantically coherent paragraphs (~350 chars/75 tokens each).  
5. Output only valid JSON matching the given schema.

**Notes**  
- Input may be English or French; output must be English.  
- Preserve original meaning; no extra commentary.

**Input**  
{text}
"""

TABLE_CLEANUP_PROMPT = """
## Task  
Convert a PDF‑extracted table into structured JSON for vector storage:

1. Understand table structure and context.  
2. Extract and interpret key data and patterns.  
3. Generate a concise descriptive summary, split into semantically coherent "paragraphs" (~350 chars/75 tokens each).  
4. Output **only** valid JSON matching the given schema.
5. If the table doesn't have useful data, output an empty array `[]`.

**Table Content**  
{table_data}
"""

IMAGE_CLEANUP_PROMPT = """
## Task  
Analyze the {image_extension} image given in context extracted from a PDF and output structured JSON:

1. If the image contains text:  
   - Extract and clean the text (remove noise, headers/footers) then describe it.  
2. If it contains illustrations:  
   - Describe the scene/concepts concisely.  
3. Output semantically coherent "paragraphs" (~350 chars/75 tokens maximum for each).

**Notes**  
- Input may be English or French; output must be English.  
- Preserve original meaning; no extra commentary.  
- Output **only** valid JSON matching the given schema.
"""

TEXT_CLEANUP_SYSTEM_PROMPT = """
You are a professional PDF content cleaner for academic and educational materials in frensh and english.
"""

TABLE_CLEANUP_SYSTEM_PROMPT = """
You are a professional PDF embedded tables cleaner for academic and educational materials in frensh and english.
"""

IMAGE_CLEANUP_SYSTEM_PROMPT = """
You are a professional PDF embedded images cleaner for academic and educational materials in frensh and english.
"""

OUTPUT_SCHEMA = {
  "name": "pdf_clean_paragraphs",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "paragraphs": {
        "type": "array",
        "description": "List of cleaned paragraphs (~350 characters/75 tokens each).",
        "items": {
          "type": "object",
          "properties": {
            "index": {
              "type": "integer",
              "description": "Paragraph number starting at 1.",
              "minimum": 1
            },
            "content": {
              "type": "string",
              "description": "Cleaned text of the paragraph."
            }
          },
          "required": ["index", "content"],
          "additionalProperties": False
        },
        "minItems": 1
      }
    },
    "required": ["paragraphs"],
    "additionalProperties": False
  }
}


class PromptTemplate:
    """Base class for prompt templates building"""
    def __init__(self, template: str, **kwargs):
        self.template = template
        self.prompt = self.template.format(**kwargs)
        

if __name__ == "__main__":
    # Example usage
    text = "\nDipôle R , L, C série   \n𝒁é𝒒= 𝒛𝑹+ 𝒛𝑳+ 𝒛𝑪= 𝑹+ 𝒋𝑳 −𝑗1\n𝐶  = 𝑹+ 𝒋(𝑳 −1\n𝐶 ) = 𝑍𝑒𝑗 \nAvec : \n{  \n  𝑍= √𝑅2+ (𝐿−1\n𝐶 )\n2\n𝑡𝑔𝑧=\n𝑰𝒎(𝒁é𝒒)\n𝑅𝑒(𝒁é𝒒) =\n𝐿−1\n𝐶 \n𝑅\n \n \nDipôle R , L, C parallèle  \n \n𝟏\n𝒛é𝒒\n= 𝟏𝒛𝑹\n⁄\n+ 𝟏𝒛𝑳\n⁄\n+ 𝟏𝒛𝑪\n⁄\n= 𝟏𝑹\n⁄\n+ 𝟏𝑱𝑳\n⁄\n+ 𝟏\n1\n𝑱𝐶 \n⁄\n= 𝟏𝑹\n⁄\n+ 𝟏𝑱𝑳\n⁄\n+ 𝑱𝐶 \n𝟏\n𝒛𝒆𝒒𝒖𝒊\n= 𝟏𝑹\n⁄\n−𝑱\n𝑳\n⁄\n+ 𝑱𝑪= 1\n𝑅+ 𝑗(𝐶−1\n𝐿 ) \n𝒛𝒆𝒒𝒖𝒊=\n𝟏\n1\n𝑅+ 𝑗(𝐶−1\n𝐿 )\n \n \nCe qui donne en multipliant numérateur et dénominateur par le conjugué du dénominateur : \n𝒛𝒆𝒒𝒖𝒊=\n1\n𝑅−𝑗(𝐶−1\n𝐿 )\n(1\n𝑅+ 𝑗(𝐶−1\n𝐿 )) (1\n𝑅−𝑗(𝐶−1\n𝐿 ))\n \n𝒛𝒆𝒒𝒖𝒊=\n1\n𝑅−𝑗(𝐶−1\n𝐿 )\n(1\n𝑅)\n2\n+ (𝐶−1\n𝐿 )\n2 \n"
    table_data = "Column1 | Column2\nValue1 | Value2"
    image_extension = "jpg"
    
    text_prompt = PromptTemplate(template=TEXT_CLEANUP_PROMPT, text=text).prompt
    table_prompt = PromptTemplate(template=TABLE_CLEANUP_PROMPT, table_data=table_data).prompt
    image_prompt = PromptTemplate(template=IMAGE_CLEANUP_PROMPT, image_extension=image_extension).prompt
    print("Text Prompt:\n", text_prompt)
    print("Table Prompt:\n", table_prompt)
    print("Image Prompt:\n", image_prompt)
