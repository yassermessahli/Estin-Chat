import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from table_cleanup import TableCleanup
from model import Model, ModelParams
from tmp import print_dashboard

params = ModelParams(
    host="http://localhost:11434",
    model="qwen3:8b",
    think=False,
    temperature=0.5,
    num_predict=70,  # for short table description
)
model = Model(params)

# Example table data - electrical engineering table
example_table_data = [
    ["", "𝒄𝒐𝒔()", "𝒔𝒊𝒏()", "𝒂 = 𝒁𝒄𝒐𝒔()", "𝒃 = 𝒁𝒔𝒊𝒏()", "𝒛 = 𝒁𝒆𝒋", "𝒆𝒋"],
    ["0", "1", "0", "𝑍", "0", "𝑍", "1"],
    ["𝜋\n2", "0", "+1", "0", "𝑍", "𝑗𝑍", "𝑗"],
    ["𝜋\n−\n2", "0", "-1", "0", "−𝑍", "−𝑗𝑍", "−𝑗"],
]

# Context from the document
example_context = """
Propriétés des opérations entre les nombres complexes  
Soit deux nombres complexes : 𝒛𝟏=𝒂𝟏+𝒋𝒃𝟏=𝒁𝟏𝒆𝒋𝟏 et 𝒛𝟐=𝒂𝟐+𝒋𝒃𝟐=𝒁𝟐𝒆𝒋𝟐 
Les nombres complexes obéissent aux mêmes règles de calcul que celles effectuées sur 
les nombres réels (addition, soustraction, multiplications et division). On obtient ainsi les 
relations suivantes.
"""

cleaner = TableCleanup(example_table_data, example_context, model)
result = cleaner.process()

# For debugging
print_dashboard(result)

# Save results
results_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, "table_description.txt")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result.message.content)

print(f"\nTable description saved to: {output_file}")
