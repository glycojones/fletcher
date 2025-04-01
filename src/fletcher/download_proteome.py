def return_proteome_url(proteome_id):
  url = f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/{proteome_id}.tar"
  return url

def return_output_path(proteome_id):
  output_path = f"/content/{proteome_id}.tar"
  return output_path

def download_proteome(proteome_id):
   url = return_proteome_url(proteome_id)
   output_path = return_output_path(proteome_id)
   if not os.path.exists(output_path):
    print("Downloading the proteome...")
    wget.download(url, output_path)
   else:
        print(proteome_id, "Proteome already downloaded!")

