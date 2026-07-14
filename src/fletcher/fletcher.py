###############################################################
# importing packages & set-up
###############################################################

import os
import argparse
import gzip
import json
import pickle
import itertools
from pathlib import Path
from math import atan2, degrees, exp, sqrt

import gemmi
import numpy as np


DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), 'data')
LIBRARY_PATH = os.path.join(DATA_DIR_PATH, 'library.gz')
library_data = None

fletcher_description = '\nFletcher will try to find a list of residues within a fixed distance from the last atom in the first residue.'\
                                '\nConcept: Federico Sabbadin & Jon Agirre, University of York, UK.'\
                                '\nCode: Jon Agirre, with contributions from Rebecca Taylor, University of York, UK.'\
                                '\n\nLatest source code: https://github.com/glycojones/fletcher\n'

###############################################################
# defining sub functions
###############################################################

CHI_ATOMS = [ { ('N', 'CA', 'CB', 'CG') : ('ARG', 'ASN', 'ASP', 'GLN', 'GLU', 'HIS', 'LEU', 'LYS',
                                           'MET', 'PHE', 'PRO', 'TRP', 'TYR', 'MSE'),
                ('N', 'CA', 'CB', 'CG1') : ('ILE', 'VAL'),
                ('N', 'CA', 'CB', 'SG') : ('CYS'),
                ('N', 'CA', 'CB', 'SE') : ('SEC'),
                ('N', 'CA', 'CB', 'OG') : ('SER'),
                ('N', 'CA', 'CB', 'OG1') : ('THR') },
              { ('CA', 'CB', 'CG', 'CD') : ('ARG', 'GLN', 'GLU', 'LYS', 'PRO'),
                ('CA', 'CB', 'CG', 'CD1') : ('LEU', 'PHE', 'TRP', 'TYR'),
                ('CA', 'CB', 'CG', 'OD1') : ('ASN', 'ASP'),
                ('CA', 'CB', 'CG', 'ND1') : ('HIS'),
                ('CA', 'CB', 'CG1', 'CD1') : ('ILE'),
                ('CA', 'CB', 'CG', 'SD') : ('MET'),
                ('CA', 'CB', 'CG', 'SE') : ('MSE') },
              { ('CB', 'CG', 'CD', 'OE1') : ('GLN', 'GLU'),
                ('CB', 'CG', 'CD', 'NE') : ('ARG'),
                ('CB', 'CG', 'CD', 'CE') : ('LYS'),
                ('CB', 'CG', 'SD', 'CE') : ('MET'),
                ('CB', 'CG', 'SE', 'CE') : ('MSE') },
              { ('CG', 'CD', 'NE', 'CZ') : ('ARG'),
                ('CG', 'CD', 'CE', 'NZ') : ('LYS') },
              { ('CD', 'NE', 'CZ', 'NH1') : ('ARG') } ]

def subtract(xyz1, xyz2):
    return [ xyz1[0] - xyz2[0], xyz1[1] - xyz2[1], xyz1[2] - xyz2[2] ]

def product(x):
    result = 1
    for x_i in x:
        result *= x_i
    return result

def dot_product(xyz1, xyz2):
    return xyz1[0] * xyz2[0] + xyz1[1] * xyz2[1] + xyz1[2] * xyz2[2]


def cross_product(xyz1, xyz2):
    return [ xyz1[1] * xyz2[2] - xyz1[2] * xyz2[1],
             xyz1[2] * xyz2[0] - xyz1[0] * xyz2[2],
             xyz1[0] * xyz2[1] - xyz1[1] * xyz2[0] ]

def magnitude(xyz):
  return (xyz[0]**2 + xyz[1]**2 + xyz[2]**2) ** 0.5

def unit(xyz):
    length = magnitude(xyz)
    return [ xyz[0] / length, xyz[1] / length, xyz[2] / length ]

def torsion(xyz1, xyz2, xyz3, xyz4, range_positive=False):
    b1 = subtract(xyz2, xyz1)
    b2 = subtract(xyz3, xyz2)
    b3 = subtract(xyz4, xyz3)
    n1 = cross_product(b1, b2)
    n2 = cross_product(b2, b3)
    m1 = cross_product(n1, n2)
    y = dot_product(m1, unit(b2))
    x = dot_product(n1, n2)
    result = degrees(atan2(y, x))
    if range_positive and result < 0:
        result += 360
    elif not range_positive and result > 180:
        result -= 360
    return result


def calculate_chis(residue):
    chis = [ ]
    for i in range(5):
        chi_atoms = [ ]
        has_chi = any(residue.name in residues for residues in list(CHI_ATOMS[i].values()))
        if not has_chi:
            return chis
        required_atom_names = next(atoms for atoms, residues in CHI_ATOMS[i].items() if residue.name in residues)
        missing_atom_names = [ ]
        for required_atom_name in required_atom_names:
            found = False
            for atom in residue:
                atom_name = atom.name.replace(' ', '')
                if atom_name in (required_atom_name, required_atom_name + ':A'):
                    chi_atoms.append(atom)
                    found = True
            if not found:
                missing_atom_names.append(required_atom_name)
        if len(chi_atoms) < 4:
            chis.append(None)
            continue
        xyzs = [ (atom.pos.x, atom.pos.y, atom.pos.z) for atom in chi_atoms ]
        chis.append(torsion(xyzs[0], xyzs[1], xyzs[2], xyzs[3]))
    return tuple(chis)


def unpack_bytes(in_bytes):
    try:
        masks = np.array([ 0b11000000, 0b00110000, 0b00001100, 0b00000011 ])
        shifts = np.array([ 6, 4, 2, 0 ])
        masked = np.array(in_bytes).reshape(-1, 1) & np.array(masks)
        shifted = masked >> np.array(shifts)
        unpacked = shifted.flatten().astype('int8')
    except ImportError:
        # Python-only mode
        bits = list(itertools.product(range(4), repeat=4))
        replaced = [ bits[b] for b in in_bytes ]
        unpacked = [ bits for byte in replaced for bits in byte ]
    return unpacked


def load_rotamer_data():
        with gzip.open(LIBRARY_PATH, 'rb') as infile:
            dim_offsets, dim_bin_ranges, dim_bin_widths, dim_num_options, compressed_byte_arrays = pickle.load(infile)
        classifications = { }
        for code, compressed in compressed_byte_arrays.items():
            compressed = bytearray(compressed)
            classifications[code] = unpack_bytes(compressed)
        global library_data
        library_data = (dim_offsets, dim_bin_ranges, dim_bin_widths, dim_num_options, classifications)

load_rotamer_data()

def get_classification(code, chis):
    global library_data
    dim_offsets, dim_bin_ranges, dim_bin_widths, dim_num_options, classifications = library_data
    if  None in chis:
        return
    if code not in dim_offsets.keys():
        return
    closest_values = [ ]
    chis = tuple([ x for x in chis if x is not None ][:len(dim_offsets[code])])
    for dimension, chi in enumerate(chis):
        dim_width = dim_bin_ranges[code][dimension][1] - dim_bin_ranges[code][dimension][0]
        if chi <= dim_bin_ranges[code][dimension][0]:
            chi += dim_width
        if chi >= dim_bin_ranges[code][dimension][1]:
            chi -= dim_width
        multiple = round((chi - dim_offsets[code][dimension]) / dim_bin_widths[code][dimension])
        closest_value = dim_offsets[code][dimension] + multiple * dim_bin_widths[code][dimension]
        closest_values.append(closest_value)
    closest_values = tuple(closest_values)
    index = 0
    for dimension, chi in enumerate(closest_values):
        dim_offset = dim_offsets[code][dimension]
        dim_bin_width = dim_bin_widths[code][dimension]
        index += int((chi - dim_offset) / dim_bin_width * product(dim_num_options[code][dimension+1:]))
    return classifications[code][index]


def plddt_to_rmsd ( plddt = 0.0 ) :
  frac_lddt = plddt / 100.0
  rmsd_estimation = 1.5 * exp(4.0*(0.7-frac_lddt))
  return rmsd_estimation


def plddt_to_bfact ( plddt = 0.0 ) :
  return min ( 999.99, 26.318945069571623 * (plddt_to_rmsd ( plddt ))**2)

def convert_residue_info_list_to_dict_format(residue_info_list):
    residue_info_list_of_dict = []
    for group in residue_info_list:
        subgroup = []
        items = group.split('~')
        for item in items:
            parts = item.strip().split(':')
            one_letter = parts[0].strip().upper()
            rotamer = parts[1] if len(parts) > 1 else 'None'
            three_letter = gemmi.expand_one_letter(one_letter, gemmi.ResidueKind.AA)
            subgroup.append({'name': three_letter, 'rotamer': rotamer})
        residue_info_list_of_dict.append(subgroup)
    return residue_info_list_of_dict

def create_script_file ( filename = "", list_of_hits = [ ] ) :
    with open ( filename.split('.')[0] + '.py', 'w' ) as file_out :
        file_out.write ( "# File programmatically created by Fletcher\n" )
        file_out.write ( 'handle_read_draw_molecule_with_recentre ("%s", 1)\n' % filename )
        file_out.write ( 'interesting_things_gui ("Results from Fletcher",[\n')
        for hit in list_of_hits:
            if hit and isinstance(hit[0], list):
                flat_residues = [item for sublist in hit for item in sublist]
            else:
                flat_residues = hit
            for residue_info in flat_residues:
                file_out.write ( '["%s %s", %.3f, %.3f, %.3f, ]' \
                                            % ( residue_info['name'], \
                                                residue_info['seqid'], \
                                                residue_info['coordinates'][0], \
                                                residue_info['coordinates'][1], \
                                                residue_info['coordinates'][2] ))
                if residue_info != flat_residues[-1] :
                    file_out.write(',\n')
            if hit != list_of_hits[-1] :
                file_out.write(',\n')
        file_out.write ( '])\n')
        file_out.close ( )

###############################################################
# main search function
###############################################################

def find_structural_motifs ( filename = "",
                             residue_info_list = [],
                             distance = 0.0,
                             min_plddt = 70.0
                             ) :
    
    af_model = gemmi.read_structure ( filename )

    neighbour_search = gemmi.NeighborSearch ( af_model[0], af_model.cell, distance ).populate ( include_h=False )
      
    residue_info_list_of_dict = convert_residue_info_list_to_dict_format(residue_info_list)

    first_residues = gemmi.Selection ( '(' + residue_info_list_of_dict[0][0]['name'] + ')' )

    list_of_hits = []

    for model in first_residues.models(af_model):
        for chain in first_residues.chains(model):
            for residue in first_residues.residues(chain):
                first_residue_info = {
                    'name': residue.name,
                    'seqid': str(residue.seqid),
                    'rotamer': str(get_classification(residue.name, calculate_chis(residue))),
                    'plddt': ('LOW PLDDT: %.2f' % residue[-1].b_iso) if residue[-1].b_iso < min_plddt else ('%.2f' % residue[-1].b_iso),
                    'coordinates': residue[-1].pos.tolist()
                }

                if first_residue_info['rotamer'] == residue_info_list_of_dict[0][0]['rotamer'] or 'None' in (first_residue_info['rotamer'], residue_info_list_of_dict[0][0]['rotamer']):

                    hit = [[] for _ in residue_info_list_of_dict]
                    hit[0].append(first_residue_info)

                    # If there are no target positions, the hit is already complete
                    if len(residue_info_list_of_dict) == 1:
                        hit_string = json.dumps(hit, sort_keys=False, indent=2)
                        if hit_string not in list_of_hits:
                            list_of_hits.append(hit)
                        continue  # Skip neighbour search – no targets to find

                    marks = neighbour_search.find_neighbors(residue[-1], 0, distance)

                    for mark in marks:
                        cra = mark.to_cra(af_model[0])

                        candidate = {
                            'name': cra.residue.name,
                            'seqid': str(cra.residue.seqid.num),  
                            'rotamer': str(get_classification(cra.residue.name, calculate_chis(cra.residue))),
                            'plddt': ('LOW PLDDT: %.2f' % cra.residue[-1].b_iso) if cra.residue[-1].b_iso < min_plddt else ('%.2f' % cra.residue[-1].b_iso),
                            'coordinates': cra.residue[-1].pos.tolist()
                        }

                        for i, entry in enumerate(residue_info_list_of_dict[1:], start=1):
                            if not hit[i]: 
                                for entry_entry in entry:
                                    if candidate['name'] == entry_entry['name']:
                                        if candidate['rotamer'] == entry_entry['rotamer'] or \
                                        'None' in (candidate['rotamer'], entry_entry['rotamer']):
                                            if all(candidate not in h for h in hit):
                                                hit[i].append(candidate)
                                                if all(len(slot) > 0 for slot in hit):

                                                    hit_string = json.dumps(hit, sort_keys = False, indent=2)
                                                    if hit_string not in list_of_hits:
                                                        list_of_hits.append(hit)    
                                                        break 

    for idx, hit in enumerate(list_of_hits, start=1):
        print(f"Hit {idx}:\n{json.dumps(hit, sort_keys = False, indent=2)}\n")                                                  
    print("Number of hits:", len(list_of_hits), "\n")

    result_dict = { }

    if len ( list_of_hits ) > 0 :
        Path ( filename ).touch() 
        result_dict['filename'] = filename
        result_dict['residue_lists'] = str(residue_info_list)
        result_dict['distance'] = distance
        result_dict['plddt'] = min_plddt
        result_dict['number of hits'] = len(list_of_hits)
        result_dict['hits'] = list_of_hits

        with open (filename.split('.')[0] + '.json', 'w' ) as file_out:
            json.dump (result_dict, file_out, sort_keys=False, indent=4)

        create_script_file (filename, list_of_hits)

    else :
        print ("\nNo results found :-( \n")
    return list_of_hits

###############################################################
# helper functions for multi‑motif parsing and filtering
###############################################################

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

###############################################################
# argument parser
###############################################################

if __name__ == '__main__':
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

  parser.add_argument ( '--motifs', \
                        help = "Multiple motifs separated by '|'. Each motif: anchor,targets:distance. "
                               "Use ',' for AND between positions, and '~' for OR between residues.", \
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
        print(f"\n✅ Structure contains ALL motifs!")
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