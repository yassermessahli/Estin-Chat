# Clean, Summarize and Standardize text using ollama models

import json
import ollama
import dataclasses


@dataclasses.dataclass
class ModelParams:
    def __init__(self):
        self.model: str = "qwen3:8b"
        self.temperature: float = 0.1
        self.sys_prompt: str = ""


class Model:
    def __init__(self, params: ModelParams):
        self.params = params

    def chat(self, prompt):
        response = ollama.chat(
            model=self.params.model, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()


class TextCleanup:
    def __init__(self, document: dict, model: Model):
        self.document = document
        self.model = model
        self.prompt = self._build_prompt()

    def _build_prompt(self):
        tables_formatted = "\n\n".join(
            [
                "\n".join([", ".join(map(str, row)) for row in table["data"]])
                for table in self.page_data.get("tables", [])
            ]
        )

        return f"prompt template to write ..."

    def _invoke_llm(self):
        return self.model.chat(self.prompt)

    def _generate_json(self, clean_content):
        try:
            return json.loads(clean_content)
        except json.JSONDecodeError:
            return {"page": self.page_data["page"], "clean content": clean_content}

    def process(self):
        llm_output = self._invoke_llm()
        return self._generate_json(llm_output)


def main():
    with open("input.json") as f:
        data = json.load(f)

    model = Model()
    cleaned_pages = [TextCleanup(page, model).process() for page in data]

    with open("output.json", "w") as f:
        json.dump(cleaned_pages, f, indent=2)

    print(f"Processed {len(cleaned_pages)} pages → output.json")


if __name__ == "__main__":
    main()
