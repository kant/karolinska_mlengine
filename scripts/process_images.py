from src.lib.imgops import create_weight_file
from src.lib.fileops import get_all_files

for ifile in get_all_files("/home/adamf/data/Karolinska/labels"):
    create_weight_file(ifile, "/home/adamf/data/Karolinska/weights_10_25")
