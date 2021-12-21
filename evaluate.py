import argparse
from vocab import Vocabulary

# import evaluation
import evaluation as evaluation

from model_CVSE import CVSE


'''1) Evaluate COCO'''
parser = argparse.ArgumentParser(description='CVSE evaluate')
parser.add_argument('--data_path', default='./Bottom_up_atten_feature/data', help='path to dataset ')
parser.add_argument('--data_name', default='coco_precomp', help='{coco,f30k}_precomp')
parser.add_argument('--concept_name', default='data/coco_annotations/Concept_annotations/category_concepts.json',
                        help='load the input concrete words of concepts')
parser.add_argument('--model_path', default='./runs/coco/CVSE_COCO/model_best.pth.tar',help='Path to load the model.')
parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary json files.')
parser.add_argument('--data_name_vocab', default='coco_precomp', help='{coco,f30k}_precomp')
parser.add_argument('--transfer_test', action='store_true', help='Whether to perform cross-dataset testing.')
parser.add_argument('--split', default='testall', help='Choose to evaluate on coco 1k test set or 5k test set. (test | testall)')
parser.add_argument('--fold5', action='store_false', help='Choose to evaluate on coco 5fold 1k test set or 5k test set')


'''2) Evaluate f30k'''
# parser = argparse.ArgumentParser(description='CVSE evaluate')
# # parser.add_argument('--data_path', default='./Bottom_up_atten_feature/data', help='path to dataset ')
# parser.add_argument('--data_name', default='f30k_precomp',   help='{coco,f30k}_precomp')
# parser.add_argument('--concept_name', default='data/f30k_annotations/Concept_annotations/category_concepts.json',
#                         help='load the input concrete words of concepts')
# parser.add_argument('--model_path', default='./runs/f30k/CVSE_f30k/model_best.pth.tar',help='Path to load the model.')    # reported in paper !
# parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary json files.')
# parser.add_argument('--data_name_vocab', default='f30k_precomp',   help='{coco,f30k}_precomp')
# parser.add_argument('--transfer_test', action='store_true', help='Whether to perform cross-dataset testing.')
# parser.add_argument('--split', default='test', help='Evaluate on f30k 1k test set. ')
# parser.add_argument('--fold5', action='store_true', help='Choose to evaluate on f30k 1k test set')


'''3) Evaluate coco-to-f30k transfer'''
# parser = argparse.ArgumentParser(description='CVSE evaluate')
# parser.add_argument('--data_path', default='./Bottom_up_atten_feature/data', help='path to dataset ')
# parser.add_argument('--data_name', default='f30k_precomp',   help='{coco,f30k}_precomp')
# parser.add_argument('--concept_name', default='data/coco_to_f30k_annotations/Concept_annotations/category_concepts.json',
#                         help='load the input concrete words of concepts') 
# parser.add_argument('--model_path', default='./runs/coco/CVSE_COCO/model_best.pth.tar',help='Path to load the model.')
# parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary json files.')
# parser.add_argument('--data_name_vocab', default='coco_precomp',   help='{coco,f30k}_precomp')
# parser.add_argument('--transfer_test', action='store_false', help='Whether to perform cross-dataset testing.')
# parser.add_argument('--concept_path', default='data/coco_to_f30k_annotations/Concept_annotations/',  
#                         help='path to load the concept data') 



def main_test():
    global args
    args = parser.parse_args()

    if args.transfer_test != True:
        evaluation.evalrank(model_path=args.model_path, data_path=args.data_path, data_name=args.data_name,
                        data_name_vocab=args.data_name_vocab, split=args.split, fold5=args.fold5, VSE_model=CVSE,
                        concept_name=args.concept_name)
    else:
        evaluation.evalrank(model_path=args.model_path, data_path=args.data_path, data_name=args.data_name,
                           data_name_vocab=args.data_name_vocab, split="test",  VSE_model=CVSE,  
                           concept_path=args.concept_path,
                           transfer_test=True,
                           concept_name=args.concept_name)

if __name__ == '__main__':
    main_test()
