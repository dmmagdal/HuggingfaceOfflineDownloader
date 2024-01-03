import os
import shutil
#import datasets
from datasets import load_dataset


wikitext = {
	"wikitext-103-v1": ['train', 'test', 'validation'],
	"wikitext-2-v1": ['train', 'test', 'validation'],
}

for rev, splits in wikitext.items():
	save_dir = os.path.join("wikitext", rev)
	cache_dir = save_dir + "_tmp"
	os.makedirs(cache_dir, exist_ok=True)
	os.makedirs(save_dir, exist_ok=True)

	for split in splits:
		save_dir = os.path.join(save_dir, split)
		os.makedirs(save_dir, exist_ok=True)
		dataset = load_dataset("wikitext", rev, split=split)
		#dataset = load_dataset("wikitext", rev, split=split, cache_dir=cache_dir)
		dataset.save_to_disk(save_dir)

	shutil.rmtree(cache_dir)

# Load dataset locally with load_from_disk() function from datasets
# module. Pass in the path to the save directory.
# https://huggingface.co/docs/datasets/v2.15.0/en/process#save
# https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/loading_methods#datasets.load_dataset
