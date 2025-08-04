## Contribute

### Requirements for contributing

We use black as our code formatter, with line length 88 (default of black). Please install black and format your files accordingly by running `python -m black .`. The easiest way is to adapt the  `settings.json` in VSCode to format automatically, with the following `settings.json`:
```
{ 
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.rulers": [
          88
        ],
      },
      "black-formatter.args": [
        "--line-length",
        "88"
    ]
}
```

### How to add a new 3D representation

* **Implement dataset class** following [BaseDataset](unifi3d/data/base_dataset.py): The dataset class takes the object of a specific dataset as input, e.g. a ShapenetPreprocessedIterator, which stores the paths to the meshes (and some preprocessed data such as SDFs, pointcloud etc). The dataset class needs to implement the `__getitem__` method where the desired files are loaded (e.g. sdf_grid.npz), transformed to the representation, and returned as tensor
* **Add representation/dataset config** in the `configs/data/` folder, which provides the configuration for initializing the dataset. (See for example [sdf](configs/data/sdf.yaml))
* **Add EncoderDecoder class** following [BaseEncoderDecoder](unifi3d/models/autoencoders/base_encoder_decoder.py): Most representations are combined with a specific autoencoder (or similar model, e.g. VAE) to encode the representation in a latent vector. The EncoderDecoder class implements the `encode` and `decode` methods to transform to the latent and back. It should be able to load a pretrained model when providing the argument `ckpt_path`.
* **Add encoder decoder config** This requires two configs: The model specifics (e.g. n_layers) should be defined in a file in `configs/net_encode`. In addition, you need to add a wrapper file in `configs/model`: This config file allows hydra to initialize an [EncoderDecoderTrainer](unifi3d/trainers/encoder_decoder_trainer.py) with your EncoderDecoder model as the `net` argument.
* **Add training config** in `configs/experiment`: Follow the structure in `configs/experiment/acc_dualoctree_ae.yaml` to specify optimizer, learning rate etc for training the encoder-decoder model on your representation
