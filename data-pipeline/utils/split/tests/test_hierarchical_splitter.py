import time
import sys
import os

# This allows the script to be run directly and find the hierarchical_splitter module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hierarchical_splitter import HierarchicalSplitter


sample_text = """
The exponential form:  
𝑧 = 𝑍𝑒^{𝑗𝜙}  
Some remarkable relations  
  
Complex conjugate  
The complex conjugate of 𝑧 (𝑧 = 𝑎 + 𝑗𝑏 = 𝑍𝑒^{𝑗𝜙}) is defined as follows:  
𝑧 = 𝑎 − 𝑗𝑏 = 𝑍𝑒^{-𝑗𝜙}  
Properties of operations between complex numbers  
Let two complex numbers: 𝑧₁ = 𝑎₁ + 𝑗𝑏₁ = 𝑍₁𝑒^{𝑗𝜙₁} and 𝑧₂ = 𝑎₂ + 𝑗𝑏₂ = 𝑍₂𝑒^{𝑗𝜙₂}  
Complex numbers obey the same calculation rules as those performed on real numbers (addition, subtraction, multiplication and division). We obtain the following relations  

Addition (or subtraction)  
𝑧 = 𝑧₁ + 𝑧₂ then  
𝑧 = 𝑎₁ + 𝑗𝑏₁ + 𝑎₂ + 𝑗𝑏₂ = (𝑎₁ + 𝑎₂) + 𝜑(𝑏₁ + 𝑏₂)  
To add (or subtract) two complex numbers, it is preferable to use the Cartesian notation.  
Product  
𝑧 = 𝑧₁ ∗ 𝑧₂ then  
𝑧 = 𝑍₁𝑒^{𝑗𝜙₁} ∗ 𝑍₂𝑒^{𝑗𝜙₂} = 𝑍₁ ∗ 𝑍₂ ∗ 𝑒^{𝑗(𝜙₁ + 𝜙₂)}  
To calculate the product of two complex numbers, it is preferable to use the polar notation.  
The modulus of the product is equal to the product of the moduli.
"""

print("\nOriginal Text:")
print(f"'{sample_text}'")
print("-" * 20)

splitter = HierarchicalSplitter()

start_time = time.time()
chunks = splitter.split_text(sample_text)
end_time = time.time()

execution_time = end_time - start_time

print(f"text splitted into {len(chunks)} chunks.")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(f"'{chunk}'")
    print(f"Length: {len(chunk)}")

print(f"Total execution time: {execution_time:.6f} seconds")
