import os

# Create gcloud command
cmd = "python -m src.combined_predict"
cmd = "%s \\\n\t%s" % (cmd, "--watershed_model %s" % "/home/adamf/data/Karolinska/watershed_7/simple/predict_model/1519293038/")
cmd = "%s \\\n\t%s" % (cmd, "--mask_model %s" % "/home/adamf/data/models/simple_predict/1519206303")
cmd = "%s \\\n\t%s" % (cmd, "--image_folder %s" % "/home/adamf/data/Karolinska/images")
cmd = "%s \\\n\t%s" % (cmd, "--output_location %s" % "/home/adamf/data/Karolinska/watershed_predicted")

os.system(cmd)