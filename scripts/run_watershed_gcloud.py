"""Create a command that can be used to run a gcloud job from terminal."""
import datetime

job_id = "adamf_%s" % datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = "adamf_20180222_102709"
project_id = 'karolinska-188312'
package_path = "src"
module_name = "src.trainer_watershed.task"
bucket = "gs://datasets-simone"
job_dir = "%s/%s" % (bucket, job_id)
runtime_version = "1.4"
region = "europe-west1"
config = "config/ml_config.yaml"
data_dir = "%s/tfrecords_512_energy" % bucket
output_dir = "%s/%s" % (bucket, job_id)
train_steps = 10000
model_type = 'medium'
batch_size = 1
save_checkpoints_steps = 100
eval_delay_secs = 5

# Create gcloud command
cmd = "gcloud ml-engine --project %s jobs submit training %s" % (project_id, job_id)
cmd = "%s \\\n\t%s" % (cmd, "--package-path %s" % package_path)
cmd = "%s \\\n\t%s" % (cmd, "--module-name %s" % module_name)
cmd = "%s \\\n\t%s" % (cmd, "--staging-bucket %s" % bucket)
cmd = "%s \\\n\t%s" % (cmd, "--job-dir %s" % job_dir)
cmd = "%s \\\n\t%s" % (cmd, "--runtime-version %s" % runtime_version)
cmd = "%s \\\n\t%s" % (cmd, "--region %s" % region)
cmd = "%s \\\n\t%s" % (cmd, "--config %s" % config)
cmd = "%s \\\n\t%s" % (cmd, "--stream-logs")
cmd = "%s \\\n\t%s" % (cmd, "--")
cmd = "%s \\\n\t%s" % (cmd, "--batch_size %s" % batch_size)
cmd = "%s \\\n\t%s" % (cmd, "--eval_delay_secs %s" % eval_delay_secs)
cmd = "%s \\\n\t%s" % (cmd, "--data_dir %s" % data_dir)
cmd = "%s \\\n\t%s" % (cmd, "--output_dir %s" % output_dir)
cmd = "%s \\\n\t%s" % (cmd, "--train_steps %s" % train_steps)
cmd = "%s \\\n\t%s" % (cmd, "--model_type %s" % model_type)
cmd = "%s \\\n\t%s" % (cmd, "--save_checkpoints_steps %s" % save_checkpoints_steps)
cmd = "%s \\\n\t%s" % (cmd, "--export_only %s" % 1)
print(cmd)
