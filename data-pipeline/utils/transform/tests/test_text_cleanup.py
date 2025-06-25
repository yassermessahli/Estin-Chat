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

example_raw_text = "L’intégral de la deuxième partie égale zéro ∫\ncos(2(  𝑡+𝑢))\n2\nT\n0\n= 0.  c’est la valeur \nmoyenne d’une grandeur alternative. \nDonc  \n𝑈𝑒𝑓𝑓\n2 = 𝑈𝑀\n2\n𝑇[t\n2]\n0\n𝑇\n= 𝑈𝑀\n2\n𝑇[T\n2 −0] \n= 𝑈𝑀\n2\n𝑇\n𝑇\n2 = 𝑈𝑀\n2\n2  \nDe cette façon on a pu démontrer l’expression de la valeur efficace :  \n𝑈𝑒𝑓𝑓= √𝑈𝑀\n2\n2\n= 𝑈𝑀\n√2\n \nN.B : la tension efficace fournie aux prises murales intérieures en Algérie est 𝑉𝑒𝑓𝑓=\n230 𝑉 à une fréquence 𝑓 = 50𝐻𝑧. \nExemple  \n𝑢(𝑡) =  14.14sin(378  𝑡+ 0.52) \nDe cette fonction on peut déduire :  \n= 378 𝑟𝑎𝑑/𝑠 \n𝑓= 𝜔\n2𝜋= 60 𝐻𝑧⇒𝑇= 1\n𝑓= 16.66 𝑚𝑠 \n𝑢= 0.52 𝑟𝑎𝑑 ⟶ 𝑢= 300   \n𝑈𝑀= 14.14 ⟶ 𝑈𝑒𝑓𝑓= 𝑈𝑒𝑓𝑓\n√2\n= 10 𝑉 \n𝑢= ∆𝑡. 𝜔= 0.52 𝑟𝑎𝑑⇒∆𝑡=\n𝑢\n𝜔= 0.52\n378 = 0.0013𝑠 \nLOIS D’OHM EN COURANT ALTERNATIF SINUSOÏDAL. \nLes lois d’Ohm s’appliquent au courant alternatif sinusoïdal. Elles s’expriment, à \nchaque instant3, dans le cas d’éléments simples, comme suit : \n \n    𝑢𝐴−𝑢𝑏= 𝑅 𝑖                                   𝑢𝐴−𝑢𝑏= 𝐿\n𝑑𝑖\n𝑑𝑡                             𝑢𝐴−𝑢𝑏=\n𝑞\n𝑐=\n1\n𝑐∫𝑖 𝑑𝑡 \n"


cleaner = TextCleanup(example_raw_text, model)
result = cleaner.process()
print_dashboard(result)


results_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(results_dir, exist_ok=True)
output_file = os.path.join(results_dir, "text_cleaned.txt")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result.message.content)

print(f"\nCleaned content saved to: {output_file}")
