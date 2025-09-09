import math
import json
import os
import gemmi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment

# ----------------- Chemistry groups -----------------
chemical_groups = {
    "hydrophobic": {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"},
    "aromatic": {"PHE", "TYR", "TRP"},
    "polar_uncharged": {"SER", "THR", "CYS", "ASN", "GLN"},
    "positive": {"LYS", "ARG", "HIS"},
    "negative": {"ASP", "GLU"},
    "special": {"GLY"}
}

# ------------------- Modifications -------------------

# Add common post-translational modifications if needed

# ----------------- lDDT-like scoring -----------------

def chemical_similarity(res1, res2):
    if res1 == res2:
        return 0.0
    for group in chemical_groups.values():
        if res1 in group and res2 in group:
            return 1.0
    return 5.0

def euclidean_distance(coord1, coord2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

def compute_lddt_like(ref_neighbours, query_neighbours,
                      thresholds=[0.5, 1.0, 2.0, 4.0], min_pairs_required=3):
    ref_pairs = {(i, j): euclidean_distance(r1['coordinates'], r2['coordinates'])
                 for i, r1 in enumerate(ref_neighbours)
                 for j, r2 in enumerate(ref_neighbours) if i < j}
    query_pairs = {(i, j): euclidean_distance(q1['coordinates'], q2['coordinates'])
                   for i, q1 in enumerate(query_neighbours)
                   for j, q2 in enumerate(query_neighbours) if i < j}

    n_pairs = min(len(ref_pairs), len(query_pairs))
    if n_pairs < min_pairs_required:
        return None, n_pairs

    cost_matrix = np.array([
        [
            euclidean_distance(r['coordinates'], q['coordinates']) +
            chemical_similarity(r['identity'], q['identity'])
            for q in query_neighbours
        ]
        for r in ref_neighbours
    ])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {r_idx: q_idx for r_idx, q_idx in zip(row_ind, col_ind)}

    fractions = []
    for t in thresholds:
        count_within = 0
        for (i, j), ref_dist in ref_pairs.items():
            if i in mapping and j in mapping:
                q_dist = euclidean_distance(
                    query_neighbours[mapping[i]]['coordinates'],
                    query_neighbours[mapping[j]]['coordinates']
                )
                if abs(q_dist - ref_dist) <= t:
                    count_within += 1
        fractions.append(count_within / len(ref_pairs))
    score = sum(fractions) / len(fractions)

    return score, n_pairs

# ----------------- Main functions -----------------

def get_reference_neighbours(ref_file_path, target_chain_id, target_res_id, distance_cutoff=15.0):
    
    if ref_file_path.endswith('.pdb'):
        ref_structure = gemmi.read_structure(ref_file_path)
    else: 
        raise ValueError("Unsupported file format. Use PDB files.")
    ref_search = gemmi.NeighborSearch(ref_structure[0], ref_structure.cell, distance_cutoff).populate(include_h=False)

    target_residue_name = None
    ref_neighbours = []

    for chain in ref_structure[0]:
        if chain.name == target_chain_id:
            for residue in chain:
                if residue.seqid.num == target_res_id:
                    target_residue_name = residue.name
                    ca_atom = next((atom for atom in residue if atom.name == 'CA'), None)
                    if ca_atom:
                        marks = ref_search.find_neighbors(ca_atom, 0, distance_cutoff)
                        for mark in marks:
                            cra = mark.to_cra(ref_structure[0])
                            if cra.atom.name == 'CA':
                                pos = cra.atom.pos
                                ref_neighbours.append({
                                    'identity': cra.residue.name,
                                    'coordinates': (pos.x, pos.y, pos.z)
                                })
                    break
    if target_residue_name is None:
        raise ValueError("Target residue not found in reference structure")

    return target_residue_name, ref_neighbours

def process_single_file(file_path, target_residue_name, ref_neighbours,
                        distance_cutoff, lddt_thresholds, min_pairs_required):
    try:
        query_structure = gemmi.read_structure(file_path)
        query_search = gemmi.NeighborSearch(query_structure[0], query_structure.cell, distance_cutoff).populate(include_h=False)

        query_candidates = []
        for chain in query_structure[0]:
            for residue in chain:
                if residue.name != target_residue_name:
                    continue
                ca_atom = next((atom for atom in residue if atom.name == 'CA'), None)
                if ca_atom:
                    marks = query_search.find_neighbors(ca_atom, 0, distance_cutoff)
                    neighbours = []
                    for mark in marks:
                        cra = mark.to_cra(query_structure[0])
                        if cra.atom.name == 'CA':
                            pos = cra.atom.pos
                            neighbours.append({
                                'identity': cra.residue.name,
                                'coordinates': (pos.x, pos.y, pos.z)
                            })
                    query_candidates.append({
                        'match_residue': {
                            'chain': chain.name,
                            'res_id': residue.seqid.num
                        },
                        'neighbours': neighbours
                    })

        results = []
        for candidate in query_candidates:
            score, n_pairs = compute_lddt_like(ref_neighbours, candidate['neighbours'],
                                               thresholds=lddt_thresholds,
                                               min_pairs_required=min_pairs_required)
            if score is not None:
                results.append({
                    'match_residue': candidate['match_residue'],
                    'score': score,
                    'n_pairs': n_pairs
                })

        if results:
            best = max(results, key=lambda x: x['score'])
            return {
                'file': os.path.basename(file_path),
                'match_residue': best['match_residue'],
                'score': best['score'],
                'n_pairs': best['n_pairs'],
                'all_results': results
            }
        else:
            return {
                'file': os.path.basename(file_path),
                'match_residue': None,
                'score': None,
                'n_pairs': 0,
                'all_results': []
            }

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return {
            'file': os.path.basename(file_path),
            'match_residue': None,
            'score': None,
            'n_pairs': 0,
            'error': str(e),
            'all_results': []
        }

def compare_query_to_reference(
        target_residue_name, 
        ref_neighbours, 
        query_path,
        distance_cutoff=15.0, 
        min_pairs_required=3,
        lddt_thresholds=[0.5, 1.0, 2.0, 4.0],
        save_results=True, 
        results_dir='results', 
        plot=True
        ):

    os.makedirs(results_dir, exist_ok=True)

    if os.path.isdir(query_path):
        query_files = [os.path.join(query_path, f) for f in os.listdir(query_path)
                       if f.endswith('.pdb')]
    elif os.path.isfile(query_path):
        if not query_path.endswith('.pdb'):
            raise ValueError("Unsupported file format. Use PDB files.")
        query_files = [query_path]
    else: 
        raise ValueError("query_path must be a directory or a PDB file.")
    
    all_file_results = []

    # Parallel processing of files
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_single_file,
                file_path,
                target_residue_name,
                ref_neighbours,
                distance_cutoff,
                lddt_thresholds,
                min_pairs_required
            ) for file_path in query_files
        ]

        for future in as_completed(futures):
            result = future.result()
            all_file_results.append(result)

    # Post-processing: saving results and plotting
    for result in all_file_results:
        if save_results and result['all_results']:
            top_results = sorted(result['all_results'], key=lambda x: x['score'], reverse=True)[:10]
            out_json_path = os.path.join(
                results_dir, 
                f"{os.path.splitext(result['file'])[0]}_lddt_top_results.json"
            )
            with open(out_json_path, 'w') as f:
                json.dump({
                    "title": "Top 10 results",
                    "score_type": "lDDT-like (chemistry-aware)",
                    "thresholds": lddt_thresholds,
                    "min_pairs_required": min_pairs_required,
                    "results": top_results
                }, f, indent=4)

            if plot:
                scores = [r['score'] for r in result['all_results']]
                sns.histplot(scores, kde=True, bins=10, color='blue')
                plt.title(f"lDDT-like scores\nQuery: {result['file']}")
                plt.xlabel("lDDT-like score (0–1)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                out_png_path = os.path.join(results_dir, f"{os.path.splitext(result['file'])[0]}_score_hist.png")
                plt.savefig(out_png_path, dpi=300, bbox_inches='tight')
                plt.close()

    valid_results = [r for r in all_file_results if r['score'] is not None]
    top_10_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)[:10]

    print("\nTop 10 query files:")
    for r in top_10_results:
        print(f"▶ {r['file']}: Best match at chain {r['match_residue']['chain']} "
              f"residue {r['match_residue']['res_id']}, score={r['score']:.4f}, pairs={r['n_pairs']}")

    if save_results and top_10_results:
        top10_json_path = os.path.join(results_dir, "top_10_file_results.json")
        with open(top10_json_path, 'w') as f:
            json.dump({
                "title": "Top 10 best scoring query files",
                "score_type": "lDDT-like (chemistry-aware)",
                "thresholds": lddt_thresholds,
                "min_pairs_required": min_pairs_required,
                "results": top_10_results
            }, f, indent=4)

    if plot and valid_results:
        all_scores = [r['score'] for r in valid_results]
        plt.figure(figsize=(6, 4))
        sns.histplot(all_scores, kde=True, bins=10, color='green')
        plt.title("Distribution of lDDT-like Scores (Best per File)")
        plt.xlabel("lDDT-like Score")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "all_files_score_histogram.png"), dpi=300)
        plt.close()

        all_pairs = [r['n_pairs'] for r in valid_results]
        plt.figure(figsize=(6, 4))
        sns.histplot(all_pairs, kde=False, bins=10, color='purple')
        plt.title("Distribution of Number of Pairs (Best per File)")
        plt.xlabel("Number of Pairs")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "all_files_pairs_histogram.png"), dpi=300)
        plt.close()
