import json

# Load class_idx_to_species_id.json
with open('class_idx_to_species_id.json', 'r', encoding='utf-8') as f:
    idx_to_species_id = json.load(f)

# Load plantnet300K_species_id_2_name.json
with open('plantnet300K_species_id_2_name.json', 'r', encoding='utf-8') as f:
    species_id_to_name = json.load(f)

# Merge: index -> species name
def build_index_to_species_name(idx_to_species_id, species_id_to_name):
    index_to_name = {}
    for idx, species_id in idx_to_species_id.items():
        name = species_id_to_name.get(species_id, None)
        index_to_name[idx] = name
    return index_to_name

index_to_species_name = build_index_to_species_name(idx_to_species_id, species_id_to_name)

# Save merged mapping
with open('index_to_species_name.json', 'w', encoding='utf-8') as f:
    json.dump(index_to_species_name, f, ensure_ascii=False, indent=2)

print('Merged mapping saved to index_to_species_name.json')
