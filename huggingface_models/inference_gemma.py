# flan_t5_raw_inference.py
# Build a simple bot with Flan T5 to test out its text generation and
# prompting capabilities.
# Windows/MacOS/Linux
# Python 3.10


import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM


def main():
	# Determine device (cpu, mps, or cuda).
	device = 'cpu'
	if torch.backends.mps.is_available():
		device = 'mps'
	elif torch.cuda.is_available():
		device = 'cuda'

	# Model ID (variant of model to load).
	# model_id = "google/flan-t5-small"
	# model_id = "google/flan-t5-base"
	# model_id = "google/flan-t5-large"			# Can run on 8GB memory (may OOM if default generate() parameters are changed)
	# model_id = "google/flan-t5-xl"			# Requires 16GB memory to run
	model_id = "google/gemma-2b"
	# model_id = "EleutherAI/gpt-neo-1.3B"
	# model_id = "EleutherAI/gpt-neo-2.7B"
	model_id = "facebook/opt-1.3b"
	# model_id = "microsoft/phi-2"
	# model_id = "RWKV/rwkv-4-3b-pile"
	# model_id = "state-spaces/mamba-2.8b"
	# model_id = "stabilityai/stablelm-3b-4e1t"
	# model_id = "stabilityai/stablelm-zephyr-3b"
	# model_id = "stabilityai/stablelm-2-1_6b"
	# model_id = "stabilityai/stablelm-2-zephyr-1_6b"
	# model_id = "xlnet-large-cased"
	# model_id = "xlnet-base-cased"
	# model_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
	if not os.path.exists(".env"):
		print("Path to .env file with huggingface token was not found")
		exit(1)
	with open(".env", "r") as f:
		token = f.read()

	# Notes:
	# Flan-t5-xl (3B) can run on the server with half precision (float15)
	# Phi-2 will OOM after one query (even at half precision)
	# GPT-Neo 2.7B will OOM even with half precision
	# GPT-Neo 1.3B can run fine at full precision
	# OPT 1.3B can run fine at full precision
	# Gemma 2B can run at half precision
	# RWKV 3B does not give good output at all (gibberish)
	# mamba-2.8b is not currently on the AutoModelForCausalLM pretrained list
	# StableLM 3B 4ELT can run with half precision
	# StableLM 3B Zephyr can run with half precision
	# StableLM 2 1.6B can run fine at full precision
	# StableLM 2 Zephyr 1.6B can run with half precision
	# XLNet large can run fine at full precision
	# XLNet base can run fine at full precision
	# TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T can run fine at full precision

	
	# Initialize tokenizer & model.
	# tokenizer = T5Tokenizer.from_pretrained(model_id)
	# model = T5ForConditionalGeneration.from_pretrained(model_id)
	tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)#, device_map="auto")
	#model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
	#model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16)
	#model = AutoModelForCausalLM.from_pretrained(model_id, token=token)
	model = AutoModelForCausalLM.from_pretrained(model_id)
	#model = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype=torch.float16)

	# Pass model to device.
	model = model.to(device)

	# Infinite loop. Prompt the model.
	print("Ctrl + C or enter \"exit\" to end the program.")
	text_input = ''
	while text_input != "exit":
		# Take in the input text.
		text_input = input("> ")
		if text_input == "exit":
			continue

		# Tokenize and process the text in the model. Print the output.
		input_ids = tokenizer(
			text_input, 
			return_tensors='pt'
		).input_ids.to(device)
		# output = model.generate(input_ids, max_length=512)
		output = model.generate(
			input_ids, 
			do_sample=True,
			min_length=64,				# default 0, the min length of the sequence to be generated (corresponds to input_prompt + max_new_tokens)
			max_length=512,				# default 20, the max langth of the sequence to be generated (corresponds to input_prompt + max_new_tokens)
			length_penalty=2,			# default 1.0, exponential penalty to the length that is used with beam based generation
			temperature=0.7,			# default 1.0, the value used to modulate the next token probabilities
			num_beams=16, 				# default 4, number of beams for beam search
			no_repeat_ngram_size=3,		# default 3
			early_stopping=True,		# default False, controls the stopping condition for beam-based methods
		)	# more detailed configuration for the model generation parameters. Depending on parameters, may cause OOM. Should play around with them to get desired output.
		print(tokenizer.decode(output[0], skip_special_tokens=True))

	# Notes:
	# -> Flan-T5 works best when prompted when doing 
	#	ConditionalGeneration. For instance: "what does George Bush
	#	do?" will return "president of united states" or "continue 
	# 	the text: the quick brown fox" gives "was able to run away from
	#	the tiger.". As you can see, the responses are quite short. 
	#	It's not *ideal* for a ChatGPT alternative but seems quite 
	#	capable.
	# -> Passing custom parameters to the model.generate() function
	#	yields much more detailed output. The downside is that it
	#	requires tuning and these parameters cannot be adjusted
	#	"on the fly" in a live application unless using a jupyter 
	#	notebook.
	# -> Using AutoTokenizer and AutoModelForSeq2SeqLM give no warning
	#	messages when initializing the model compared to using the
	#	T5Tokenizer and T5ModelForConditionalGeneration classes.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
