"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions
from chemprop.train import transfer_train
import json
from argparse import Namespace

# python predict.py --test_path Whitehead_results/No_Morgan_counts/fold_0/test_full.csv --checkpoint_path Whitehead_results/No_Morgan_counts/fold_0/model_0/model.pt --preds_path Whitehead_results/Predictions/no_morgan_counts.csv

# python predict.py --test_path Anti_nCoV_results/Scaffold_split/fold_0/test_full.csv --checkpoint_path Anti_nCoV_results/Scaffold_split/fold_0/model_0/model.pt --preds_path Anti_nCoV_results/Scaffold_split/Predictions/morgan_count_predictions.csv --features_generator morgan_count

# if __name__ == '__main__':
#     my_args = json.load(open('predict_args.json','r'))
#     args = Namespace(**my_args)
#     make_predictions(args)



def transfer_learn_from_model(checkpoint_dir, tune_data_path, test_data_path, lr_ratio = 0.1, layer_list = None):
	# print(checkpoint_dir)
	# print(preds_path)

	transfer_train.run_transfer_training(checkpoint_dir, tune_data_path, test_data_path, lr_ratio = lr_ratio, layer_list = layer_list)

def get_base_predict_args():
	base_predict_args = json.load(open('base_predict_args.json','r'))
	return Namespace(**base_predict_args)


# def transfer_learn_from_model(preds_dir, base = True, morgan = True, rdkit = True, morgan_plus_rdkit = True, test_data_path = None):
# 	predictions_dir = preds_dir + '/'
# 	if is_in_vivo:
# 		predictions_dir += 'in_vivo_'
# 	predict_from_json(get_base_predict_args(), preds_path = predictions_dir+'Predictions/base_only.csv',
# 		checkpoint_dir = preds_dir + '/Base_model_only', test_data_path = test_data_path)

# preds_dir = 'Whitehead_results/in_vitro_only/Amine_split_500s'
# layer_list = [1]*20+[2]*20+[3]*20+[4]*20+[5]*20
# for model in ['/Base_model_only','/Morgan_counts','/Morgan_plus_rdkit','/rdkit_2d']:
# 	transfer_learn_from_model(checkpoint_dir = preds_dir + model,
# 		tune_data_path = 'Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo/tune.csv',
# 		test_data_path = 'Data/Whitehead_in_vivo_splits/Amine_split_500s_in_vivo/test.csv',
# 		lr_ratio = 1, layer_list = layer_list)



