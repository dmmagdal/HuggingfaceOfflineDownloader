# ONNX Converter

Description: This folder exists to convert models from Huggingface (usually PyTorch) to ONNX format.


### Notes

 - There is a distinct difference between using the `optimum-cli` (from the `optimum` package) and the conversion script provided by `transformers.js`. 
     - If you intend to deploy your models to JS with `transformers.js`, set up a virtual environment (`conda` or `virtualenv`) with the packages specified in the `requirements.txt` in the `transformers.js` repo and use the conversion script in the repo (command: `python scripts/convert.py --mode_id model_id_or_path --quantize`). The `--quantize` flag is optional in case you want to quantize the resulting model. 
     - If you simply wish to convert the model to ONNX, use the conversion command from `optimum` (command: `optimum-cli export onnx --model_id model_id_or_path output_folder`).
 - It's also possible to run the necessary conversions from a docker image (`dockerfile` included in the repo). Use the following steps below to set up the environment:
     - 


### Converted models

#### ONNX (optimum-cli)

 - Flan-T5
     - flan-t5-small (80M) [model](https://huggingface.co/dmmagdal/flan-t5-small-onnx)
     - flan-t5-base (250M) [model](https://huggingface.co/dmmagdal/flan-t5-base-onnx)
     - flan-t5-large (780M) [model](https://huggingface.co/dmmagdal/flan-t5-large-onnx)
     - flan-t5-xl (3B) [model](https://huggingface.co/dmmagdal/flan-t5-xl-onnx)
     - flan-t5-xxl (11B) [model](https://huggingface.co/dmmagdal/flan-t5-xxl-onnx)
 - Whisper
     - whisper-tiny (39M) [model](https://huggingface.co/dmmagdal/whisper-tiny-onnx)
     - whisper-base (74M) [model](https://huggingface.co/dmmagdal/whisper-base-onnx)
     - whisper-small (244M) [model](https://huggingface.co/dmmagdal/whisper-small-onnx)
     - whisper-medium (769M) [model](https://huggingface.co/dmmagdal/whisper-medium-onnx)
     - whisper-large (1.5B) [model](https://huggingface.co/dmmagdal/whisper-large-onnx)
     - whisper-large-v2 (1.5B) [model](https://huggingface.co/dmmagdal/whisper-large-v2-onnx)
     - whisper-large-v3 (1.5B) [model](https://huggingface.co/dmmagdal/whisper-large-v3-onnx)


#### transformers.js

 - GPT 2
     - gpt2 (124M) [regular](https://huggingface.co/dmmagdal/gpt2-onnx-js) [quantized](https://huggingface.co/dmmagdal/gpt2-onnx-js-quantized)
     - gpt2-medium (355M) [regular](https://huggingface.co/dmmagdal/gpt2-medium-onnx-js) [quantized](https://huggingface.co/dmmagdal/gpt2-medium-onnx-js-quantized)
     - gpt2-large (774M) [regular](https://huggingface.co/dmmagdal/gpt2-large-onnx-js) [quantized](https://huggingface.co/dmmagdal/gpt2-large-onnx-js-quantized)
     - gpt2-xl (1.5B) [regular](https://huggingface.co/dmmagdal/gpt2-xl-onnx-js) [quantized](https://huggingface.co/dmmagdal/gpt2-xl-onnx-js-quantized)
     - distilgpt2 (82M) [regular](https://huggingface.co/dmmagdal/distilgpt2-onnx-js) [quantized](https://huggingface.co/dmmagdal/distilgpt2-onnx-js-quantized)
 - GPT Neo
     - gpt-neo-125M [regular](https://huggingface.co/dmmagdal/gpt-neo-125M-onnx-js) [quantized](https://huggingface.co/dmmagdal/gpt-neo-125M-onnx-js-quantized)
     - gpt-neo-1.3B [regular](https://huggingface.co/dmmagdal/gpt-neo-1.3B-onnx-js) [quantized](https://huggingface.co/dmmagdal/gpt-neo-1.3B-onnx-js-quantized)
     - gpt-neo-2.7B [regular](https://huggingface.co/dmmagdal/gpt-neo-2.7B-onnx-js) (quantized not available)
 - OPT
     - opt-125M [regular](https://huggingface.co/dmmagdal/opt-125m-onnx-js) [quantized](https://huggingface.co/dmmagdal/opt-125m-onnx-js-quantized)
     - opt-350M [regular](https://huggingface.co/dmmagdal/opt-350m-onnx-js) [quantized](https://huggingface.co/dmmagdal/opt-350m-onnx-js-quantized)
     - opt-1.3B [regular](https://huggingface.co/dmmagdal/opt-1.3b-onnx-js) [quantized](https://huggingface.co/dmmagdal/opt-1.3b-onnx-js-quantized)
     - opt-2.7B [regular](https://huggingface.co/dmmagdal/opt-2.7b-onnx-js) (quantized not available)
 - Flan-T5
     - flan-t5-small (80M) [regular](https://huggingface.co/dmmagdal/flan-t5-small-onnx-js) [quantized](https://huggingface.co/dmmagdal/flan-t5-small-onnx-js-quantized)
     - flan-t5-base (250M) [regular](https://huggingface.co/dmmagdal/flan-t5-base-onnx-js) [quantized](https://huggingface.co/dmmagdal/flan-t5-base-onnx-js-quantized)
     - flan-t5-large (780M) [regular](https://huggingface.co/dmmagdal/flan-t5-large-onnx-js) [quantized](https://huggingface.co/dmmagdal/flan-t5-large-onnx-js-quantized)
     - flan-t5-xl (3B) [regular](https://huggingface.co/dmmagdal/flan-t5-xl-onnx-js) [quantized](https://huggingface.co/dmmagdal/flan-t5-xl-onnx-js-quantized)
     - flan-t5-xxl (11B) (regular not available) (quantized not available)


### References

 - [Export a model to ONNX with optimum.exporters.onnx](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)
 - [Convert your models to ONNX with transformers.js](https://huggingface.co/docs/transformers.js/custom_usage#convert-your-models-to-onnx)
 - Google Models Releases
     - [Google BERT release](https://huggingface.co/collections/google/bert-release-64ff5e7a4be99045d1896dbc)
     - [Google T5 release](https://huggingface.co/collections/google/t5-release-65005e7c520f8d7b4d037918)
     - [Google Flan-T5 release](https://huggingface.co/collections/google/flan-t5-release-65005c39e3201fff885e22fb)
     - [Google Switch Transformer release](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f)
     - [Google MT5 release](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f)