import os

def parse_xyz_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        num_atoms = int(lines[0].strip())
        atom_types = set()

        for line in lines[2:]:  # Skip the first two lines
            parts = line.split()
            atom_type = parts[0]
            atom_types.add(atom_type)

        return num_atoms, atom_types

def process_directory(directory):
    max_atoms = 0
    all_atom_types = set()

    for filename in os.listdir(directory):
        if filename.endswith('.xyz'):
            filepath = os.path.join(directory, filename)
            num_atoms, atom_types = parse_xyz_file(filepath)
            max_atoms = max(max_atoms, num_atoms)
            all_atom_types.update(atom_types)

    return max_atoms, all_atom_types

# Specify the directory containing the .xyz files
directory = 'qm7'

max_atoms, all_atom_types = process_directory(directory)

print(f"Maximum number of atoms in a single file: {max_atoms}")
print(f"Types of atoms found across all files: {', '.join(sorted(all_atom_types))}")

