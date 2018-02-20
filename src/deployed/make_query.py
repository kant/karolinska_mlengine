import googleapiclient.discovery
import numpy as np
import os

from src.lib.imgops import plot_image_from_array, image_to_base64_websafe_resize, load_image

class MakeQuery():
    def __init__(self, project, model, version=None, client_secret=None):
        # Set the environment variable
        if client_secret:
            secret_path = client_secret
        else:
            secret_path = os.path.abspath('./config/client_secret.json')

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = secret_path

        # Create service object and prepare the name of the model we are going to query
        self._service = googleapiclient.discovery.build('ml', 'v1')
        self._name = 'projects/{}/models/{}'.format(project, model)

        if version is not None:
            self._name += '/versions/{}'.format(version)

    def predict(self, instances):
        response = self._service.projects().predict(name=self._name, body=instances).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        return response

    def get_prediction(self, image_files):
        input_list = [{'input':
                      image_to_base64_websafe_resize(img, (512, 512))} for img in image_files]
        input_format = {'instances': input_list}

        return self.predict(input_format)


if __name__ == "__main__":
    # TODO: Add argparse
    image_test = '/home/adamf/data/Karolinska/images/Nuclei_22.png'
    mq = MakeQuery('karolinska-188312', 'karolinska', 'v3')
    response = mq.get_prediction([image_test])
    mask_response = response['predictions'][0]['classes']
    org_image, _ = load_image(image_test, size=(512, 512))

    def apply_tinge(original, prediction):
        # Load and make the predicted cells a bit more red
        original = np.stack([original.copy()] * 3, 2)
        x_idx, y_idx = np.where(prediction)
        original[x_idx, y_idx, 1] /= 5
        original[x_idx, y_idx, 2] /= 5
        plot_image_from_array(original, show_type='matplotlib')

    # Simple argmax prediction
    apply_tinge(org_image, mask_response)

    # More strict predictions
    mask_probability = np.array(response['predictions'][0]['probabilities'])
    for i_level in [0.25, 0.5, 0.75]:
        apply_tinge(org_image, mask_probability[:, :, 1] > i_level)
