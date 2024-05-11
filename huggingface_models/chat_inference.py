# chat_inference.py
# Simple script that runs a text generation (or text2text generation)
# on chat models for inference. This primarily is to test out a model's
# out of the box prompting and generation capabilities.
# Windows/MacOS/Linux
# Python 3.10


import os
import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM
from transformers import pipeline


# NOTE:
# Sources on chat templating and using text generation chat models
# https://huggingface.co/docs/transformers/main/en/chat_templating
# https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
# https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T


def main():
	# Determine device (cpu, mps, or cuda).
	device = 'cpu'
	if torch.backends.mps.is_available():
		# device = 'mps' # mps device causes errors and more memory usage for tinyllama while cpu doesnt
		device = 'cpu'
	elif torch.cuda.is_available():
		device = 'cuda'

	# Model ID (variant of model to load).
	# model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
	# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
	model_id = "./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0"
	if not os.path.exists(".env"):
		print("Path to .env file with huggingface token was not found")
		exit(1)
	with open(".env", "r") as f:
		token = f.read()
	
	# Initialize tokenizer & model.
	# tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)#, device_map="auto")
	#model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
	#model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16)
	#model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
	# model = AutoModelForCausalLM.from_pretrained(model_id)
	#model = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype=torch.float16)
	pipe = pipeline("text-generation", model_id, token=token, device=device)

	# Pass model to device.
	# model = model.to(device)

	# Initial chat template to pass into the model
	messages = [
		{
			"role": "system",
			"content": "You are a friendly chatbot who always responds to any query or question asked",
		},
		{
			"role": "user",
			"content": "",
		}
	]

	# Infinite loop. Prompt the model.
	print("Ctrl + C or enter \"exit\" to end the program.")
	text_input = ''
	while text_input != "exit":
		# Take in the input text.
		text_input = input("> ")
		if text_input == "exit":
			continue

		# Insert user text input to user content.
		messages[-1]["content"] = text_input

		# Apply the prompt as a chat template. Pass the template prompt
		# to the model and print the output.
		prompt = pipe.tokenizer.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True
		)
		outputs = pipe(
			prompt, 
			max_new_tokens=1024,
			do_sample=True,
			temperature=0.7,
			# top_k=50, # from tinyllama chat v1.0 model card. Gave "RuntimeError: Currently topk on mps works only for k<=16"
			top_k=8,
			top_p=0.95
		)
		print(outputs[0]["generated_text"])

		# Tokenize and process the text in the model. Print the output.
		# input_ids = tokenizer(
		# 	text_input, 
		# 	return_tensors='pt'
		# ).input_ids.to(device)
		# # output = model.generate(input_ids, max_length=512)
		# output = model.generate(
		# 	input_ids, 
		# 	do_sample=True,
		# 	min_length=64,				# default 0, the min length of the sequence to be generated (corresponds to input_prompt + max_new_tokens)
		# 	max_length=512,				# default 20, the max langth of the sequence to be generated (corresponds to input_prompt + max_new_tokens)
		# 	length_penalty=2,			# default 1.0, exponential penalty to the length that is used with beam based generation
		# 	temperature=0.7,			# default 1.0, the value used to modulate the next token probabilities
		# 	num_beams=16, 				# default 4, number of beams for beam search
		# 	no_repeat_ngram_size=3,		# default 3
		# 	early_stopping=True,		# default False, controls the stopping condition for beam-based methods
		# )	# more detailed configuration for the model generation parameters. Depending on parameters, may cause OOM. Should play around with them to get desired output.
		# print(tokenizer.decode(output[0], skip_special_tokens=True))

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
