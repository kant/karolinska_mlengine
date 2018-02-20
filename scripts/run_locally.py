"""Wrapper to run a training job locally and changing parameters to the training job."""
import os

if __name__ == "__main__":
    # Training command
    local = True
    cmd = "python -m src.trainer.task"
    if local:
        cmd = "%s %s" % (cmd, "--data_dir /home/adamf/data/Karolinska/tfrecords_512")
        cmd = "%s %s" % (cmd, "--output_dir /home/adamf/data/Karolinska/model_4")
    else:
        cmd = "%s %s" % (cmd, "--data_dir /hdd/Karolinska/tfrecords_512")
        cmd = "%s %s" % (cmd, "--output_dir /hdd/users/adamf/models/Karolinska/attempt_21")
    cmd = "%s %s" % (cmd, "--batch_size 2")
    cmd = "%s %s" % (cmd, "--train_steps %s" % 25000)
    cmd = "%s %s" % (cmd, "--model_type %s" % 'simple')
    cmd = "%s %s" % (cmd, "--save_checkpoints_steps %s" % 500)
    cmd = "%s %s" % (cmd, "--save_summary_steps %s" % 2)
    raise
    os.system(cmd)
