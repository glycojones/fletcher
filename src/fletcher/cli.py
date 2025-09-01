import argparse

import os
import sys

# Get the absolute path to the parent of this file (the "src" directory)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from .fletcher import get_reference_neighbours, compare_query_to_reference

def main():
    parser = argparse.ArgumentParser(
        prog = "Fletcher",
        description = "Chemistry-aware lDDT-like scoring for PDB files",
        epilog = "Example usage: fletcher --ref path/to/ref.pdb --queries path/to/query1.pdb path/to/query2.pdb --target_chain A --target_res 100")
    
    parser.add_argument('-r', '--ref', help = "Reference PDB path", required = True)
    
    parser.add_argument('-q', '--query', \
                        help = "Query PDB file", \
                        required = True)
    
    parser.add_argument('-c', '--target_chain', \
                        help = "Target chain ID in reference structure", \
                        required = True)
    
    parser.add_argument('-t', '--target_res', \
                        help = "Target residue number in reference structure",
                        type = int, 
                        required = True)
    
    parser.add_argument('--results_dir', \
                        default = "results", \
                        help="Directory to save results")
    
    parser.add_argument('--distance_cutoff', \
                        type=float, 
                        default=15.0, 
                        help="Distance cutoff for neighbors")
    
    parser.add_argument('--min_pairs_required', \
                        type=int,
                        default=3, 
                        help="Minimum number of CA-atom pairs required to compute a score (default: 3)")
    
    parser.add_argument('--lddt_thresholds', \
                        nargs='+', 
                        default=[0.5, 1.0, 2.0, 4.0], 
                        type=float,
                        help="Distance thresholds for lDDT scoring (default: 0.5 1.0 2.0 4.0)")

    args = parser.parse_args()

    target_residue_name, ref_neighbours = get_reference_neighbours(
        args.ref, 
        args.target_chain,
        args.target_res,
        distance_cutoff=args.distance_cutoff
    )

    compare_query_to_reference(
        target_residue_name, 
        ref_neighbours, 
        args.query,
        distance_cutoff=args.distance_cutoff,
        min_pairs_required=args.min_pairs_required,
        lddt_thresholds=args.lddt_thresholds,
        results_dir=args.results_dir
    )

if __name__ == "__main__":
    main()

