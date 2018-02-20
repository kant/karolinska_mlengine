from src.lib.imgops import plot_image_from_array, image_to_base64_websafe_resize
import numpy as np
import os
import json
import ast


def get_prediction(model_dir, image_files):
    input_list = [{'input':
                  image_to_base64_websafe_resize(img, (480, 320))} for img in image_files]

    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")

    with open('/tmp/tmp.json', 'w') as outfile:
        json.dump(input_list[0], outfile)

    cmd = "gcloud ml-engine --verbosity debug local predict --model-dir %s --json-instances %s" % \
        (model_dir, '/tmp/tmp.json')
    output = os.popen(cmd).read()

    # For now, I manually parse the output... which is annoying.
    classes_start = output.index("[[")
    classes_end = output.index("]]  [[") + 2
    prob_start = output.index("]]  [[") + 4
    prob_end = output.index("]]]") + 3
    classes = np.array(ast.literal_eval(output[classes_start:classes_end]))
    probabilities = np.array(ast.literal_eval(output[prob_start:prob_end]))
    plot_image_from_array(classes, show_type="matplotlib")


if __name__ == "__main__":
    # TODO: Add argparse
    model_dir = "/home/adamf/data/carvana/models/v4/simple/predict_model/1517233230/"
    get_prediction(model_dir, ["/home/adamf/data/carvana/individualImage.png"])
