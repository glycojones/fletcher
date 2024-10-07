

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