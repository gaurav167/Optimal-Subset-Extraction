import os
import random
import argparse

from CONSTANTS import FULL_DATA_DIR, SAMPLE_DATA_DIR


def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def copy(filenames, source_dir, extension, dest_dir):
	for filename in filenames:
		source_file_path = os.path.join(source_dir, extension, filename)
		dest_file_path = os.path.join(dest_dir, extension, filename)
		os.popen('cp ' + source_file_path + ' ' + dest_file_path) # Can be replaced with explicit python Read-Write file commands


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=float, required=True, help='Fraction of sample size [0-1]')

args = parser.parse_args()

source_train_path = os.path.join(FULL_DATA_DIR, 'train')
source_test_path = os.path.join(FULL_DATA_DIR, 'test')
sample_train_path = os.path.join(SAMPLE_DATA_DIR, 'train')
sample_test_path = os.path.join(SAMPLE_DATA_DIR, 'test')

train_pos = os.listdir(os.path.join(source_train_path, 'pos'))
train_neg = os.listdir(os.path.join(source_train_path, 'neg'))
test_pos = os.listdir(os.path.join(source_test_path, 'pos'))
test_neg = os.listdir(os.path.join(source_test_path, 'neg'))

create_dir(os.path.join(sample_train_path, 'pos'))
create_dir(os.path.join(sample_train_path, 'neg'))
create_dir(os.path.join(sample_test_path, 'pos'))
create_dir(os.path.join(sample_test_path, 'neg'))

sample_train_pos = random.sample(population=train_pos, k=int(len(train_pos)*args.size))
sample_train_neg = random.sample(population=train_neg, k=int(len(train_neg)*args.size))
sample_test_pos = random.sample(population=test_pos, k=int(len(test_pos)*args.size))
sample_test_neg = random.sample(population=test_neg, k=int(len(test_neg)*args.size))

copy(sample_train_pos, source_train_path, 'pos', sample_train_path)
copy(sample_train_neg, source_train_path, 'neg', sample_train_path)
copy(sample_test_pos, source_test_path, 'pos', sample_test_path)
copy(sample_test_neg, source_test_path, 'neg', sample_test_path)
