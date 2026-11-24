# Eyes on the Image

Code for *Eyes on the Image: Gaze Supervised Multimodal Learning for Chest X-ray Diagnosis and Report Generation*, accepted at the FABRIC Workshop at WACV.

Install with Python 3.10 or newer.

```sh
python -m pip install ".[detector,evaluation]"
python -m mimic_eye --help
```

The pipeline prepares local MIMIC-Eye data, trains the classifier, predicts from radiographs, and generates and evaluates reports. See the [reproduction guide](docs/reproduce.md) for the paper settings, checkpoint requirements, and evaluation details.

Run the synthetic check with `python -m unittest discover -s tests`. It uses small local models and a mock report server.
 
