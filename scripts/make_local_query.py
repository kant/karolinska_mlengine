import os

# Create gcloud command
cmd = "python -m src.trainer.predict"
cmd = "%s \\\n\t%s" % (cmd, "--prediction_model %s" % "/home/adamf/data/models/simple_predict/1519206303")
cmd = "%s \\\n\t%s" % (cmd, "--image_folder %s" % "/home/adamf/data/Karolinska/images")
cmd = "%s \\\n\t%s" % (cmd, "--output_location %s" % "/home/adamf/data/Karolinska/images_predicted")

os.system(cmd)