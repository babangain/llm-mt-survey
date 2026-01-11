We use wmt14 data for our experiments, use ```python download_data_de.py``` and it should automatically download the data inside data folder. 

Run the ```python process_one_shot_near.py``` and other files, which will process the files already created within ./data folder to create few-shot examples.

All the training and testing confifgurations are in ```LLaMA-Factory/llamaboard_config```, Clone ```https://github.com/hiyouga/LlamaFactory``` and install with ```pip install -e .``` Then, copy llamaboard_config inside the folder.
Then run the following commands

```
llamafactory-cli train llamaboard_config/en-de-full.yaml
```

When the training is complete, run ```en-de-full-test.yaml``` assuming that the model checkpoints are saved into the default directory.
