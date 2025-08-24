import argparse
import glob

import os
import sys

# Get the absolute path to the parent of this file (the "src" directory)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from fletcher.fletcher import get_reference_neighbours, compare_queries_to_reference

def main():
    parser = argparse.ArgumentParser(description="Chemistry-aware lDDT-like scoring for PDB files")
    parser.add_argument("--ref", required=True, help="Reference PDB path")
    parser.add_argument("--queries", required=True, nargs="+", help="Query PDB files or glob pattern")
    parser.add_argument("--target_chain", required=True, help="Target chain ID in reference structure")
    parser.add_argument("--target_res", type=int, required=True, help="Target residue number in reference structure")
    parser.add_argument("--results_dir", default="results", help="Directory to save results")
    parser.add_argument("--distance_cutoff", type=float, default=15.0, help="Distance cutoff for neighbors")
    parser.add_argument("--min_pairs_required", default=3, type=int,help="Minimum number of CA-atom pairs required to compute a score (default: 3)")
    parser.add_argument("--lddt_thresholds", nargs='+', default=[0.5, 1.0, 2.0, 4.0], type=float,help="Distance thresholds for lDDT scoring (default: 0.5 1.0 2.0 4.0)")

    args = parser.parse_args()

    # Expand glob patterns
    query_paths = []
    for pattern in args.queries:
        query_paths.extend(glob.glob(pattern))

    target_residue_name, ref_neighbours = get_reference_neighbours(
        args.ref, 
        args.target_chain,
        args.target_res,
        distance_cutoff=args.distance_cutoff
    )

    compare_queries_to_reference(
        target_residue_name, 
        ref_neighbours, 
        args.queries,
        distance_cutoff=args.distance_cutoff,
        min_pairs_required=args.min_pairs_required,
        lddt_thresholds=args.lddt_thresholds,
        results_dir=args.results_dir
    )

if __name__ == "__main__":
    main()

