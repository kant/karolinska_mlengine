import argparse
import os

# TODO: Convert this to a bash script instead
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        help='Name of this model',
        default='Carvana'
    )
    parser.add_argument(
        '--version',
        help='Version of this model',
        default='v1'
    )
    parser.add_argument(
        '--bucket_path',
        help='Location of exported model',
        default='gs://datasets-simone/adamf_20180218_215016/unet/predict_model/1519027295/'
    )

    args = parser.parse_args()
    # cmd = "gcloud ml-engine models create %s" % args.model_name
    cmd = "gcloud ml-engine versions create %s --model %s --origin %s --runtime-version=1.4" % \
        (args.version, args.model_name, args.bucket_path)
    os.system(cmd)
