import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from text_cleanup import TextCleanup
from model import Model, ModelParams
from tmp import print_dashboard

params = ModelParams(
    host="http://localhost:11434",
    model="qwen3:8b",
    think=False,
    temperature=0.5,
)
model = Model(params)

example_raw_text = "La forme exponentielle :  \n𝒛= 𝒁𝒆𝒋 \nQuelques relations remarquables  \n \n \nComplexe conjugué  \nOn définit le complexe conjugué de 𝒛 (𝒛= 𝒂+ 𝒋𝒃= 𝒁𝒆𝒋) comme suit : \n𝒛= 𝒂−𝒋𝒃= 𝒁𝒆−𝒋 \nPropriétés des opérations entre les nombres complexes  \nSoit deux nombres complexes : 𝒛𝟏= 𝒂𝟏+ 𝒋𝒃𝟏= 𝒁𝟏𝒆𝒋𝟏 et 𝒛𝟐= 𝒂𝟐+ 𝒋𝒃𝟐= 𝒁𝟐𝒆𝒋𝟐 \nLes nombres complexes obéissent aux mêmes règles de calcul que celles effectuées sur \nles nombres réels (addition, soustraction, multiplications et division). On obtient ainsi les \nrelations suivantes  \nAddition (ou soustraction) \n𝒛= 𝒛𝟏+ 𝒛𝟐 alors  \n𝒛= 𝒂𝟏+ 𝒋𝒃𝟏+ 𝒂𝟐+ 𝒋𝒃𝟐= (𝒂𝟏+𝒂𝟐) + 𝒋(𝒃𝟏+ 𝒃𝟐) \nPour additionner (ou soustraire) deux nombres complexes on utilise de préférence \nla notation cartésienne. \nProduit  \n𝒛= 𝒛𝟏∗𝒛𝟐 alors  \n𝐳= 𝐙𝟏𝐞𝐣𝟏∗𝐙𝟐𝐞𝐣𝟐= 𝐙𝟏∗𝐙𝟐∗𝐞𝐣(𝟏+𝟐) \nPour calculer le produit de deux nombres complexes on utilise de préférence la notation \npolaire. \nLe module du produit est égal au produit des modules. \n"


cleaner = TextCleanup(example_raw_text, model)
result = cleaner.process()
print_dashboard(result)


results_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, "text_cleaned.txt")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result.message.content)

print(f"\nCleaned content saved to: {output_file}")
