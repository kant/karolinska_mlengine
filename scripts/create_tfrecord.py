"""Runs tfrecord script in a terminal shell."""
import os

# Tfrecord command
cmd = "python -m src.tfrecord.parsers"
cmd = "%s %s" % (cmd, "--output_path /home/adamf/data/Karolinska/tfrecords_512_energy")
cmd = "%s %s" % (cmd, "--feature_folder /home/adamf/data/Karolinska/images")
cmd = "%s %s" % (cmd, "--label_folder /home/adamf/data/Karolinska/labels_single")
cmd = "%s %s" % (cmd, "--weight_folder /home/adamf/data/Karolinska/weights_10_25")
cmd = "%s %s" % (cmd, "--split 0.9 0.1 0")
cmd = "%s %s" % (cmd, "--img_size 512 512")
cmd = "%s %s" % (cmd, "--shards 2")
os.system(cmd)
