import json
import os
import gemmi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

# ----------------- Chemistry groups -----------------
chemical_groups = {
    "hydrophobic": {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"},
    "aromatic": {"PHE", "TYR", "TRP"},
    "polar_uncharged": {"SER", "THR", "CYS", "ASN", "GLN"},
    "positive": {"LYS", "ARG", "HIS"},
    "negative": {"ASP", "GLU"},
    "special": {"GLY"}
}

# Create a mapping from residue to its chemical group for quick lookup
residue_to_group = {}
for group_name, residues in chemical_groups.items():
    for res in residues:
        residue_to_group[res] = group_name

def chemical_similarity(res1, res2):
    if res1 == res2:
        return 0.0
    return 1.0 if residue_to_group.get(res1) == residue_to_group.get(res2) else 5.0

# ----------------- lDDT-like scoring -----------------

def compute_score(
        ref_neighbours, 
        query_neighbours,
        lddt_thresholds, 
        min_pairs_required=3,
        all_atoms=False
        ):

    if all_atoms:
        raise NotImplementedError("All atoms scoring not implemented yet.")
    else:
        ref_coords = np.array([r['coordinates'] for r in ref_neighbours])
        ref_ids = [r['residue_identity'] for r in ref_neighbours]

        query_coords = np.array([r['coordinates'] for r in query_neighbours])
        query_ids = [r['residue_identity'] for r in query_neighbours]

        n = len(ref_coords)
        if n<2:
            return None, 0
    
        # compute reference pairwise distances
        ref_dists = pdist(ref_coords)
        ref_pairs = np.triu_indices(n, k=1)
        n_pairs = len(ref_dists)

        if n_pairs < min_pairs_required:
            return None, n_pairs
    
        # compute cost matrix using broadcasting
        cost_matrix = np.linalg.norm(ref_coords[:, None, :] - query_coords[None, :, :], axis=2)
        for i in range(n):
            for j in range(len(query_coords)):
                cost_matrix[i, j] += chemical_similarity(ref_ids[i], query_ids[j])

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        mapping = {r_idx: q_idx for r_idx, q_idx in zip(row_ind, col_ind)}

        # compute score

        fractions = []
        for t in lddt_thresholds:
            count_within = 0
            for idx, (i,j) in enumerate(zip(*ref_pairs)):
                if i in mapping and j in mapping:
                    qi = query_coords[mapping[i]]
                    qj = query_coords[mapping[j]]
                    q_dist = np.linalg.norm(qi-qj)
                    if abs(q_dist - ref_dists[idx]) <= t:
                        count_within += 1
            fractions.append(count_within / n_pairs)

        score = sum(fractions) / len(fractions)
        return score, n_pairs

# ----------------- Main functions -----------------

def get_reference_neighbours(ref_file_path, target_chain_id, target_res_id, distance_cutoff=15.0, all_atoms=False):
    
    # need to work on this so that it can read mmcif files too
    if ref_file_path.endswith('.pdb'):
        ref_structure = gemmi.read_structure(ref_file_path)
    else: 
        raise ValueError("Unsupported file format. Use PDB files.")

    ref_search = gemmi.NeighborSearch(ref_structure[0], ref_structure.cell, distance_cutoff).populate(include_h=False)
    print(f"Reference structure loaded and neighbours searched: {ref_file_path}")

    # NB read_structure takes 1 second per structure to run and so does Neighbor_Search
    # so reading & processing takes 2 seconds per structure hence the slow run time

    target_residue_name = None
    ref_neighbours = []

    for chain in ref_structure[0]:
        # isolating the target chain
        if chain.name != target_chain_id:
            continue
        # iterating through residues in the target chain to find target residue
        for residue in chain:
            if residue.seqid.num != target_res_id:
                continue
            target_residue_name = residue.name # found the target residue

            # now find neighbours of target residue
            if all_atoms:
                for atom in residue:
                    marks = ref_search.find_neighbors(atom, 0, distance_cutoff)
                    for mark in marks:
                        cra = mark.to_cra(ref_structure[0])
                        pos = cra.atom.pos
                        ref_neighbours.append({
                            'residue_identity': cra.residue.name,
                            'atom_identity': cra.atom.name,
                            'coordinates': (pos.x, pos.y, pos.z)
                        }) 
            else:
                ca_atom = next((atom for atom in residue if atom.name == 'CA'), None)
                if not ca_atom:
                    continue
                marks = ref_search.find_neighbors(ca_atom, 0, distance_cutoff)
                for mark in marks:
                    cra = mark.to_cra(ref_structure[0])
                    if cra.atom.name == 'CA':
                        pos = cra.atom.pos
                        ref_neighbours.append({
                            'residue_identity': cra.residue.name,
                            'coordinates': (pos.x, pos.y, pos.z)
                        })
            break  

    if target_residue_name is None:
        raise ValueError("Target residue not found in reference structure")

    return target_residue_name, ref_neighbours

def process_single_file(file_path, target_residue_name, ref_neighbours,
                        distance_cutoff, all_atoms, lddt_thresholds, min_pairs_required):
    try:
        query_structure = gemmi.read_structure(file_path)
        query_search = gemmi.NeighborSearch(query_structure[0], query_structure.cell, distance_cutoff).populate(include_h=False)

        print(f"{file_path} processed")

        query_candidates = []

        # find all residues in query that match target residue name
        for chain in query_structure[0]:
            for residue in chain:
                if residue.name != target_residue_name:
                    continue

                # found a candidate residue, now find its neighbours
                if all_atoms:
                    for atom in residue:
                        # find neighbours for each atom
                        marks = query_search.find_neighbors(atom, 0, distance_cutoff)
                        neighbours = []
                        for mark in marks:
                            cra = mark.to_cra(query_structure[0])
                            pos = cra.atom.pos
                            neighbours.append({
                                'residue_identity': cra.residue.name,
                                'atom_identity': cra.atom.name,
                                'coordinates': (pos.x, pos.y, pos.z)
                            })
                        query_candidates.append({
                            'match_residue': {
                                'chain': chain.name,
                                'res_id': residue.seqid.num
                            },
                            'neighbours': neighbours
                        })
                else:
                    ca_atom = next((atom for atom in residue if atom.name == 'CA'), None)
                    if not ca_atom:
                        continue
                    marks = query_search.find_neighbors(ca_atom, 0, distance_cutoff)
                    neighbours = []
                    for mark in marks:
                        cra = mark.to_cra(query_structure[0])
                        if cra.atom.name == 'CA':
                            pos = cra.atom.pos
                            neighbours.append({
                                'residue_identity': cra.residue.name,
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

        # score each candidate
        for candidate in query_candidates:
            score, n_pairs = compute_score(ref_neighbours, 
                                           candidate['neighbours'],
                                           lddt_thresholds=lddt_thresholds,
                                           min_pairs_required=min_pairs_required,
                                           all_atoms=all_atoms)
            if score is not None:
                results.append({
                    'match_residue': candidate['match_residue'],
                    'score': score,
                    'n_pairs': n_pairs
                })

        # find best scoring candidate
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

    # Error handling
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

# ----------------- Compare Query to Reference -----------------
def compare_query_to_reference(
        target_residue_name, 
        ref_neighbours, 
        query_path,
        distance_cutoff=15.0, 
        all_atoms=False,
        min_pairs_required=3,
        lddt_thresholds=[0.5, 1.0, 2.0, 4.0],
        save_results=True, 
        results_dir='results', 
        plot=True
        ):

    # prepare results directory
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # gather query files
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
    # Using ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers = os.cpu_count() or 4) as executor:
        futures = [
            executor.submit(
                process_single_file,
                file_path,
                target_residue_name,
                ref_neighbours,
                distance_cutoff,
                all_atoms,
                lddt_thresholds,
                min_pairs_required
            ) for file_path in query_files
        ]

        for future in as_completed(futures):
            result = future.result()
            all_file_results.append(result)

    # # Post-processing: saving results and plotting
    # for result in all_file_results:
    #     if save_results and result['all_results']:
    #         top_results = sorted(result['all_results'], key=lambda x: x['score'], reverse=True)[:10]
    #         out_json_path = os.path.join(
    #             results_dir, 
    #             f"{os.path.splitext(result['file'])[0]}_lddt_top_results.json"
    #         )
    #         with open(out_json_path, 'w') as f:
    #             json.dump({
    #                 "title": "Top 10 results",
    #                 "score_type": "lDDT-like (chemistry-aware)",
    #                 "thresholds": lddt_thresholds,
    #                 "min_pairs_required": min_pairs_required,
    #                 "results": top_results
    #             }, f, indent=4)

    #         if plot:
    #             scores = [r['score'] for r in result['all_results']]
    #             sns.histplot(scores, kde=True, bins=10, color='blue')
    #             plt.title(f"lDDT-like scores\nQuery: {result['file']}")
    #             plt.xlabel("lDDT-like score (0–1)")
    #             plt.ylabel("Frequency")
    #             plt.tight_layout()
    #             out_png_path = os.path.join(results_dir, f"{os.path.splitext(result['file'])[0]}_score_hist.png")
    #             plt.savefig(out_png_path, dpi=300, bbox_inches='tight')
    #             plt.close()

    # ----- Summarize top 10 results across all files -----

    # filter out files with no valid score
    valid_results = [r for r in all_file_results if r['score'] is not None]

    # sort by score and take top 10
    top_10_results = sorted(valid_results, key=lambda x: x['score'], reverse=True)[:10]

    # print summary
    print("\nTop 10 query files:")
    for r in top_10_results:
        print(f"▶ {r['file']}: Best match at chain {r['match_residue']['chain']} "
              f"residue {r['match_residue']['res_id']}, score={r['score']:.4f}, pairs={r['n_pairs']}")

    # save top 10 summary
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

    # plot overall distributions
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

        print(f"Score histogram saved to {results_dir}")

        all_pairs = [r['n_pairs'] for r in valid_results]
        plt.figure(figsize=(6, 4))
        sns.histplot(all_pairs, kde=False, bins=10, color='purple')
        plt.title("Distribution of Number of Pairs (Best per File)")
        plt.xlabel("Number of Pairs")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "all_files_pairs_histogram.png"), dpi=300)
        plt.close()

        print(f"Pairs histogram saved to {results_dir}")