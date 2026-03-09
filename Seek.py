from pymatgen.core import Structure

# Cargar CIF
structure = Structure.from_file("../Cif/BiTeCl_mp-28944_primitive.cif")

# ---------------------------
# 1. Vectores de red (Å)
# ---------------------------
lattice_vectors = structure.lattice.matrix

print("Vectores de red en coordenadas cartesianas (Å):")
print(lattice_vectors)

# ---------------------------
# 2. Posiciones atómicas cartesianas
# ---------------------------
cart_coords = structure.cart_coords

print("\nPosiciones atómicas en coordenadas cartesianas (Å):")
for i, site in enumerate(structure):
    print(f"{site.species_string} {cart_coords[i]}")