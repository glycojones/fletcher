import json
import argparse
import itertools
import numpy as np
from .fletcher import find_structural_motifs, create_script_file

###############################################################
# helper functions for multi‑motif parsing and filtering
###############################################################

def check_proximity(flat_hit, max_distance):
    if max_distance is None or max_distance <= 0:
        return True
    coords = [np.array(r['coordinates']) for r in flat_hit]
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist > max_distance:
                return False
    return True

def parse_motifs_from_string(motifs_string):
    motifs = []
    for motif_part in motifs_string.split('|'):
        motif_part = motif_part.strip()
        if not motif_part:
            continue
        if ':' in motif_part:
            residues_part, dist_str = motif_part.rsplit(':', 1)
            try:
                dist = float(dist_str)
            except ValueError:
                residues_part = motif_part
                dist = None
        else:
            residues_part = motif_part
            dist = None
        motifs.append({
            'residues': residues_part,
            'distance': dist
        })
    return motifs

def cli():
    ###############################################################
    # argument parser
    ###############################################################
    
    fletcher_description = '\nFletcher will try to find a list of residues within a fixed distance from the last atom in the first residue.'\
                                '\nConcept: Federico Sabbadin & Jon Agirre, University of York, UK.'\
                                '\nCode: Tom Robinson, Rebecca Taylor & Jon Agirre, University of York, UK.'\
                                '\n\nLatest source code: https://github.com/glycojones/fletcher\n'
    
    parser = argparse.ArgumentParser ( 
                    prog='Fletcher',
                    description=fletcher_description,
                    epilog='' )

    parser.add_argument ( '-f', '--filename', \
                            help = "The name of the file to be processed, in PDB or mmCIF format.", \
                            required = True )                  

    parser.add_argument ( '-r', '--residues', \
                            help = "A list of residues in one-letter code, comma separated, including alternatives, e.g. H:1,H:2,F:1~W:3:2~Y", \
                            default = "", required = False )                       

    parser.add_argument ( '-m', '--motifs', \
                            help = "Multiple motifs separated by '|'. Each motif: anchor,targets:distance. "
                                "Use ',' for AND between positions, and '~' for OR between residues."
                                "Can also specify rotamers as integers (check what rotamer you want using an"
                                "experimental structure first!)"
                                "Example: --motifs \"H:3,F:5.0 | H~D:5.0\""
                                "Which means a histidine in rotamer 3 and a phenyl alanine within 5 Å,"
                                "with another histidine or aspartic acid within 5 Å, separated by DISTANCE.", \
                            required = False )                       

    parser.add_argument ( '-d', '--distance', \
                            help = "Default distance if not specified per motif in --motifs (Angstroms).", \
                            default = "5.0", required = False )  

    parser.add_argument ( '--proximity', \
                            help = "Maximum allowed distance (Å) between any two residues from different motifs.", \
                            default = "0.0", required = False )

    parser.add_argument ( '-p', '--plddt', \
                            help = "Flag up candidate residues with average pLDDT below threshold (Jumper et al., 2020).", \
                            default = "70.0", required = False )
    
    parser.add_argument ( '-n', '--nterm', \
                            help = 'Require one residue to be at the n-terminus', \
                            choices = [ 'yes', 'no' ], \
                            default = 'no' )
    
    parser.add_argument ( '-c', '--cterm', \
                            help = 'Require one residue to be at the c-terminus', \
                            choices = [ 'yes', 'no' ], \
                            default = 'no' )

    args = parser.parse_args ( )

    ###############################################################
    # interpretation of arguments
    ###############################################################

    filename = args.filename
    distance = float ( args.distance )
    min_plddt = float ( args.plddt )
    proximity_threshold = float ( args.proximity ) if args.proximity else 0.0

    ###############################################################
    # multi‑motif mode
    ###############################################################

    if args.motifs:
        print ( fletcher_description )
        print ( "Running Fletcher with the following parameters:\n"
                "\nFilename: ", filename, 
                "\nMotifs: ", args.motifs, 
                "\nDefault distance: ", distance, 
                "\nProximity: ", proximity_threshold if proximity_threshold > 0 else "off",
                "\npLDDT: ", min_plddt,
                "\n" 
                )
        
        parsed_motifs = parse_motifs_from_string(args.motifs)
        if not parsed_motifs:
            print("ERROR: No valid motifs found in --motifs")
            exit(1)
        
        all_motif_hits = []
        for idx, motif in enumerate(parsed_motifs):
            motif_dist = motif['distance'] if motif['distance'] is not None else distance
            residues_list = motif['residues'].split(',')
            print(f"Searching for motif {idx+1}: '{motif['residues']}' at {motif_dist} Å")
            hits = find_structural_motifs(filename, residues_list, motif_dist, min_plddt)
            if not hits:
                print(f"❌ Motif {idx+1} NOT found. Stopping.")
                exit(0)
            all_motif_hits.append(hits)
        
        combined_hits = []
        duplicate_count = 0
        proximity_discard_count = 0
        
        for combo in itertools.product(*all_motif_hits):
            flat_hit = []
            for single_hit in combo:
                for residue_slot in single_hit:
                    flat_hit.extend(residue_slot)
            
            seen_seqids = set()
            has_overlap = False
            for residue in flat_hit:
                seqid = residue['seqid']
                if seqid in seen_seqids:
                    has_overlap = True
                    break
                seen_seqids.add(seqid)
            if has_overlap:
                duplicate_count += 1
                continue
            
            if proximity_threshold > 0:
                if not check_proximity(flat_hit, proximity_threshold):
                    proximity_discard_count += 1
                    continue
            
            combined_hits.append(flat_hit)
        
        if duplicate_count > 0:
            print(f"  ⚠️  Skipped {duplicate_count} combined hit(s) due to residue overlap.")
        if proximity_discard_count > 0:
            print(f"  ⚠️  Skipped {proximity_discard_count} combined hit(s) due to proximity ({proximity_threshold} Å).")
        
        if combined_hits:
            print("\n✅ Structure contains ALL motifs!")
            print(f"Found {len(combined_hits)} valid combined hit(s).\n")
            for idx, hit in enumerate(combined_hits, start=1):
                print(f"--- Combined Hit {idx} ---")
                print(json.dumps(hit, sort_keys=False, indent=2))
                print()
            
            result_dict = {
                'filename': filename,
                'motifs': args.motifs,
                'default_distance': distance,
                'proximity_threshold': proximity_threshold if proximity_threshold > 0 else None,
                'plddt': min_plddt,
                'number_of_combined_hits': len(combined_hits),
                'hits': combined_hits
            }
            with open (filename.split('.')[0] + '_multisite.json', 'w' ) as file_out:
                json.dump (result_dict, file_out, sort_keys=False, indent=4)
            create_script_file(filename, combined_hits)
            print(f"JSON output written to {filename.split('.')[0]}_multisite.json")
            print(f"PyMOL script written to {filename.split('.')[0]}.py")
        else:
            print("\n❌ No valid combined hits after filtering.")

    ###############################################################
    # single‑motif mode (legacy)
    ###############################################################

    elif args.residues:
        residue_info_list = args.residues.split(',')
        print ( fletcher_description )
        print ( "Running Fletcher with the following parameters:\n"
                "\nFilename: ", filename, 
                "\nResidue list: ", residue_info_list, 
                "\nDistance: ", distance, 
                "\npLDDT: ", min_plddt,
                "\n" 
                )
        find_structural_motifs(filename, residue_info_list, distance, min_plddt)

    else:
        print("ERROR: Please provide either -r (single motif) or --motifs (multi-motif).")
        exit(1)