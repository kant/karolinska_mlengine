# Tensorflow Estimators - Segmentation
This repository will contain multiple kinds of Segmentation Neural Networks, implemented in Tensorflow. The networks are wrapped within an Tensorflow Estimator which takes care of higher level functions, such as early stopping, evaluations, eventual model serving.

Using Estimators also allows training on the GCP cloud without much effort.

## Getting Started

First, you need to download a dataset. The one I have been using is found here: https://www.kaggle.com/c/carvana-image-masking-challenge
You will have to register before you can download the data.

### Prerequisites

This implementation needs

```
Python 3+
Tensorflow 1.4
gsutil - https://cloud.google.com/storage/docs/gsutil_install
```

### Prepare the data

Begin by downloading the desired dataset, unzip, and sort all images into one folder and all labels into another. Then you will create tfrecords for training with. From the root directory of the repo, I would, for example, run the following

```
python3 -m src.tfrecord.parsers \
	--output_path /home/adamf/data/carvana/tfrecords \
	--feature_folder /home/adamf/data/carvana/train_hq \
	--label_folder /home/adamf/data/carvana/train_masks \
	--split 0.8 0.1 0.1 \
	--img_size 320 480 \
	--shards 10
```

Which will start creating tfrecord files, where each image is resized to (320x480) and the number of generated tfrecords will be 10. It's better to have smaller shards than big, but if your RAM is sufficient, it is safe to decrease this number and have bigger tfrecords.

### Running locally
To run locally, simply run

```
python3 -m src.trainer.task \
	--data_dir /home/adamf/data/carvana/tfrecords \
	--output_dir /home/adamf/data/carvana/models \
	--model_type unet \
	--batch_size 5
```

You can choose between different model types, look into `src.trainer.arch` for all available models.

### Running on gcloud
To run on the cloud, you will first have to create a bucket in the desired project. In this case, I assume that the bucket name is "carvana". Then you will have to copy your tfrecords to the cloud:

```
gsutil cp /home/adamf/data/carvana/tfrecords/*.tfrecords gs://carvana/tfrecords
```

And then you can schedule an ML job (I assume that any API access has been opened). Open up a terminal and cd into the project root, then
```
gcloud ml-engine jobs submit training YOUR_UNIQUE_JOB_ID \
	--package-path src \
	--module-name src.trainer.task \
	--staging-bucket gs://carvana \
	--job-dir gs://carvana/JOB_ID \
	--runtime-version 1.4 \
	--region europe-west1 \
	--config config/ml_config.yaml \
	-- \
	--data_dir gs://carvana/tfrecords \
	--output_dir gs://carvana/JOB_ID \
	--train_steps 10000 \
	--model_type unet
```

Take a look at `scripts/run_gcloud.py`. There you can change different values and create the desired gcloud command.
To cancel a running ML job, run
```
gcloud ml-engine jobs cancel YOUR_UNIQUE_JOB_ID
```

#### Changing ML Engine specs
The ML Engine service has a number of available computer specs available. When prototyping, I would first make sure that your project will run locally at first. Then, to check if everything works okay on GCP, I would use the simplest VM available.

To change what specs to use with the project, look at `config/ml_config.yaml`, specifically the scaleTier variable.
A list of available tiers can be found here: https://cloud.google.com/ml-engine/docs/pricing
## Authors

* **Adam Fjeldsted** - *Initial work* 
