def clean_hidden_files(file_list):
    return [f for f in file_list if not f.startswith(".")]
