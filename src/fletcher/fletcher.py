import os
import gemmi
import argparse
import gzip
import json
import pickle
import itertools
import numpy as np
from pathlib import Path
from math import acos, atan2, degrees, exp

DATA_DIR_PATH = os.path.join(os.path.dirname(__file__), 'data')
LIBRARY_PATH = os.path.join(DATA_DIR_PATH, 'library.gz')
library_data = None


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
                atom_name = atom.name.replace(' ', '') # No idea if we'll need .replace 
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
        dim_offest = dim_offsets[code][dimension]
        dim_bin_width = dim_bin_widths[code][dimension]
        index += int((chi - dim_offest) / dim_bin_width * product(dim_num_options[code][dimension+1:]))
    return classifications[code][index]


def plddt_to_rmsd ( plddt = 0.0 ) :
  frac_lddt = plddt / 100.0
  rmsd_estimation = 1.5 * exp(4.0*(0.7-frac_lddt))
  return rmsd_estimation


def plddt_to_bfact ( plddt = 0.0 ) :
  return min ( 999.99, 26.318945069571623 * (plddt_to_rmsd ( plddt ))**2)


def create_script_file ( filename = "", list_of_hits = [ ] ) :
  with open ( filename.split('.')[0] + '.py', 'w' ) as file_out :
    file_out.write ( "# File programmatically created by Fletcher\n" )
    file_out.write ( 'handle_read_draw_molecule_with_recentre ("%s", 1)\n' % filename )
    file_out.write ( 'interesting_things_gui ("Results from Fletcher",[\n')
    for hit in list_of_hits :
      file_out.write ( '["%s %s", %.3f, %.3f, %.3f, ]' \
                                % ( hit[0].get('name'), \
                                    hit[0].get('seqid'), \
                                    hit[0].get('coordinates')[0], \
                                    hit[0].get('coordinates')[1], \
                                    hit[0].get('coordinates')[2] ))
      if hit is not list_of_hits[-1] :
        file_out.write(',\n')
    file_out.write ( '])\n')
    file_out.close ( )


def find_structural_motifs ( filename = "",
                             residue_lists = [ ],
                             rotamer_lists = [],
                             distance = 0.0,
                             min_plddt = 70.0,
                             n_term = False,
                             c_term = False,
                            ) :
  load_rotamer_data()
  af_model = gemmi.read_structure ( filename )
  neighbour_search = gemmi.NeighborSearch ( af_model[0], af_model.cell, distance ).populate ( include_h=False )
  first_residues = gemmi.Selection ( '(' + residue_lists[0][0] + ')' ) 
  
  result_dict = { }
  result_list = [ ]

  for model in first_residues.models(af_model):
    for chain in first_residues.chains(model):
      for residue in first_residues.residues(chain):
        partial_result = [ residue ]
        marks = neighbour_search.find_neighbors ( residue[-1], 0, distance )
        for candidate_list in residue_lists[1:] :
          for candidate in candidate_list :
            found_in_contacts = False
            for mark in marks :
              cra = mark.to_cra ( af_model[0] )
              
              # We do the following conversion to harness gemmi's translation of modified residue codes
              # into the unmodified ones, e.g. HIC (methylated histidine) >> HIS (normal histidine)
              if gemmi.find_tabulated_residue(candidate).one_letter_code.upper() == \
                 gemmi.find_tabulated_residue(cra.residue.name).one_letter_code.upper() \
                 and cra.residue not in partial_result :
                
                partial_result.append ( cra.residue )
                found_in_contacts = True
                break
            if found_in_contacts :
              break
          if len(residue_lists) == len(partial_result) :
            if (n_term or c_term) :
              in_terminus = False
              for residue in partial_result :
                if n_term and residue == chain[0] :
                  in_terminus = True
                elif c_term and residue.seqid.num == chain[-1].seqid.num :
                  in_terminus = True
              if in_terminus : result_list.append ( partial_result )
            else :
              result_list.append ( partial_result )
            
  if len ( result_list ) > 0 :
    Path ( filename ).touch() # We want results at the top
    result_dict['filename'] = filename
    result_dict['residue_lists'] = str(residue_lists)
    result_dict['distance'] = distance
    result_dict['plddt'] = min_plddt
    hit_list = []
    match_list = []

    for result in result_list :
      hit = [ ]
      for residue in result :
        residue_dict = { }
        residue_dict['name']  = residue.name
        residue_dict['seqid'] = str(residue.seqid)
        residue_dict['rotamer'] = str(get_classification(residue.name, calculate_chis(residue)))
        if residue[-1].b_iso < min_plddt :
          residue_dict['plddt'] = 'LOW PLDDT: %.2f' % residue[-1].b_iso
        else :
          residue_dict['plddt'] = '%.2f' % residue[-1].b_iso
        residue_dict ['coordinates'] = residue[-1].pos.tolist()
        hit.append ( residue_dict )
        match = filter_for_residue_and_rotamer_match(converted_output, hit)
    match_list.append(match)
    hit_list.append(hit)
    print ("Match found:", match)
    result_dict['hits'] = hit_list

    with open (filename.split('.')[0] + '.json', 'w' ) as file_out:
      json.dump (result_dict, file_out, sort_keys=False, indent=4)
    
    create_script_file (filename, hit_list)
  
  else :
    print ("\nNo results found :-( \n")
  return result_dict

if __name__ == '__main__':
  parser = argparse.ArgumentParser ( 
                    prog='Fletcher',
                    description='Fletcher will try to find a list of residues within a fixed distance from the centre of mass.'\
                                '\nConcept: Federico Sabbadin & Jon Agirre, University of York, UK.',
                    epilog='Please send bug reports to Jon Agirre: jon.agirre@york.ac.uk' )

  parser.add_argument ( '-f', '--filename', \
                        help = "The name of the file to be processed, in PDB or mmCIF format.", \
                        required = True )                  

  parser.add_argument ( '-r', '--residues', \
                        help = "A list of residues in one-letter code, comma separated, including alternatives, e.g. H:1,H:2,F:1~W:3:2~Y", \
                        default = "GF", required = True )                       

  parser.add_argument ( '-d', '--distance', \
                        help = "Specifies how far each of the residues can be from the rest, in Angstroems.", \
                        default = "0.0", required = True )  

  parser.add_argument ( '-p', '--plddt', \
                        help = "Flag up candidate residues with average pLDDT below thresold (Jumper et al., 2020).", \
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
  
  # Assuming argparse has got the right number of parameters beyond this point

  print ( "\nFletcher is a tool that helps spot and document molecular features in AlphaFold models."\
          "\nConcept: Federico Sabbaddin & Jon Agirre, University of York, UK."\
          "\nLatest source code: https://github.com/glycojones/fletcher"\
          "\nBug reports to jon.agirre@york.ac.uk\n\n" )
  
  input_criteria = args.residues.split(',')

  # input_criteria ['K:2', 'H:3', 'A~F~W:3']

  list_of_items = [
        [item.split(':') for item in slot.split('~')] 
        for slot in input_criteria
    ]

  # list_of_items [[['K', '2']], [['H', '3']], [['A'], ['F'], ['W', '3']]]

  input_residues = ["".join(entry[0] for entry in item) for item in list_of_items]
  list_of_residues = [ ]

  for slot in input_residues :
    list_of_residues.append ( gemmi.expand_one_letter_sequence(slot, gemmi.ResidueKind.AA) )

  # print("input_residues", input_residues)
  # input_residues ['K', 'H', 'AFW']
  # print("list_of_residues", list_of_residues)
  # list_of_residues [['LYS'], ['HIS'], ['ALA', 'PHE', 'TRP']]

  # choose the second object in each list within each list in list_of_items if there actually is a rotamer specified
  list_of_rotamers = [
    [[entry[1]] if len(entry) > 1 else [''] for entry in item]  
    for item in list_of_items
  ]

  # print("list_of_rotamers", list_of_rotamers)
  # list_of_rotamers [[['2']], [['3']], [[''], [''], ['3']]]

  def convert_to_nested_dict(list_of_residues, list_of_rotamers):
    result = []
    for names_group, rotamers_group in zip(list_of_residues, list_of_rotamers):
        sub_result = []
        for name, rotamer_list in zip(names_group, rotamers_group):
            rotamer = rotamer_list[-1] if rotamer_list else ''
            sub_result.append({'name': name, 'rotamer': rotamer})
        result.append(sub_result)
    return result
  
  converted_output = convert_to_nested_dict(list_of_residues, list_of_rotamers)
  # print("Converted output:", converted_output)
  # Converted output: [[{'name': 'LYS', 'rotamer': '2'}], [{'name': 'HIS', 'rotamer': '3'}], [{'name': 'ALA', 'rotamer': ''}, {'name': 'PHE', 'rotamer': ''}, {'name': 'TRP', 'rotamer': '3'}]]

  def filter_for_residue_and_rotamer_match(converted_output, potential_hit):
    match = []
    for potential_hit_list in potential_hit:
      for converted_output_list in converted_output:
        for converted_output_sublist in converted_output_list:
          if potential_hit_list['name'] == converted_output_sublist['name']:
            if potential_hit_list['rotamer'] == converted_output_sublist['rotamer'] or converted_output_sublist['rotamer'] == '':
              match.append(potential_hit_list)
    if len(match) >= len(converted_output):
      return match

  distance = float ( args.distance )
  min_plddt = float ( args.plddt )
  n_term = True if args.nterm == 'yes' else False
  c_term = True if args.cterm == 'yes' else False

  print ( "Running Fletcher with the following parameters:\n"
          "\nFilename: ", args.filename, 
          "\nResidue list: ", list_of_residues, 
          "\nRotamer list: ", list_of_rotamers, 
          "\nDistance: ", distance, 
          "\npLDDT: ", min_plddt,
          "\nN-term: ", n_term,
          "\nC-term: ", c_term,
          "\n" )

  if len ( list_of_residues ) > 1 and distance > 0.0 :
    find_structural_motifs ( args.filename, list_of_residues, list_of_rotamers, distance, min_plddt, n_term, c_term )

