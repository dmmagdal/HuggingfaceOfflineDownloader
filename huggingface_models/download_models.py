import os
import json
import shutil
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer


def main():
	# HF auth token (required for gemma).
	auth_path = ".env"
	token = False
	if os.path.exists(auth_path):
		with open(auth_path, "r") as f:
			token = f.read().strip("\n")

	# Models.
	t5 = ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",]
	bert = ["bert-base-uncased"]
	gpt_neo = ["EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-125M"]
	opt = ["facebook/opt-350m", "facebook/opt-125m", "facebook/opt-1.3b"]
	mobile_bert = ["google/mobilebert-uncased"]
	gemma = ["google/gemma-2b", "google/gemma-2b-it"]
	tiny_llama = ["TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
	llama = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"]

	# Download loops.
	for t in t5:
		save_dir = t.replace('/', '_')
		cache_dir = save_dir + "_tmp"
		os.makedirs(cache_dir, exist_ok=True)
		os.makedirs(save_dir, exist_ok=True)
		tokenizer = AutoTokenizer.from_pretrained(t, cache_dir=cache_dir)
		model = AutoModelForSeq2SeqLM.from_pretrained(t, cache_dir=cache_dir)

		#tokenizer.save_pretrained(cache_dir)
		#model.save_pretrained(cache_dir)
		tokenizer.save_pretrained(save_dir)
		model.save_pretrained(save_dir)
		shutil.rmtree(cache_dir)

	for gpt in gpt_neo + opt + gemma + tiny_llama + llama:
		# Special condition to skip the download for gemma if the 
		# auth token is not specified.
		if ("gemma" in gpt or "meta" in gpt) and not token:
			continue

		save_dir = gpt.replace('/', '_')
		cache_dir = save_dir + "_tmp"
		os.makedirs(cache_dir, exist_ok=True)
		os.makedirs(save_dir, exist_ok=True)
		tokenizer = AutoTokenizer.from_pretrained(
			gpt, cache_dir=cache_dir, use_auth_token=token
		)
		model = AutoModelForCausalLM.from_pretrained(
			gpt, cache_dir=cache_dir, use_auth_token=token
		)

		#tokenizer.save_pretrained(cache_dir)
		#model.save_pretrained(cache_dir)
		tokenizer.save_pretrained(save_dir)
		model.save_pretrained(save_dir)
		shutil.rmtree(cache_dir)

	for bert in bert + mobile_bert:
		save_dir = bert.replace('/', '_')
		cache_dir = save_dir + "_tmp"
		os.makedirs(cache_dir, exist_ok=True)
		tokenizer = BertTokenizer.from_pretrained(bert, cache_dir=cache_dir)
		model = BertModel.from_pretrained(bert, cache_dir=cache_dir)

		os.makedirs(save_dir, exist_ok=True)
		#tokenizer.save_pretrained(cache_dir)
		#model.save_pretrained(cache_dir)
		tokenizer.save_pretrained(save_dir)
		model.save_pretrained(save_dir)
		shutil.rmtree(cache_dir)

	# Load each model/tokenizer by specifying the path instead of the model_id
	# in the from_pretrained() function. You may have to set the local_files_only
	# argument to True.
		
	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()
