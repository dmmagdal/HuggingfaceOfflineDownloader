import os
import json
import shutil
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import BertModel, BertTokenizer

t5 = ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",]
bert = ["bert-base-uncased"]
gpt_neo = ["EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-125M"]
opt = ["facebook/opt-350m", "facebook/opt-125m", "facebook/opt-1.3b"]
mobile_bert = ["google/mobilebert-uncased"]

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

for gpt in gpt_neo + opt:
	save_dir = gpt.replace('/', '_')
	cache_dir = save_dir + "_tmp"
	os.makedirs(cache_dir, exist_ok=True)
	os.makedirs(save_dir, exist_ok=True)
	tokenizer = AutoTokenizer.from_pretrained(gpt, cache_dir=cache_dir)
	model = AutoModelForCausalLM.from_pretrained(gpt, cache_dir=cache_dir)

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
