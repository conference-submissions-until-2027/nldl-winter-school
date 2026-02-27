import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
import networkx as nx
import json
from trainer import SkinCancerTrainer, BreakPointOnEpochCallback
from dataset import SkinLesionDataset
import torch
from hyperbolic_model import HyperbolicConvNeXt
from euclidian_model import EuclideanConvNeXt
import numpy as np
from argparse import ArgumentParser, Namespace
import ast
from prototype_generator import HyperbolicPrototypeGenerator, generate_prototypes
from utils import get_class_weights, generate_sampler, dataset_sanity_check, generate_l1_mapping_from_dataset_and_file,generate_cascading_sampler
# Import the classes we just wrote
# from dataset import SkinLesionDataset
# from model import HyperbolicConvNeXt
# from trainer import SkinCancerTrainer

BATCH_SIZE = 32
EPOCHS = 100
EMBEDDING_DIM = 10 # Must match the dimension in prototypes.pkl



def save_meta_data_json(args, output_dir):
    
    save_path = os.path.join(output_dir, 'experiment_args.json')
    with open(save_path, 'w', encoding = 'utf-8') as f:
        json.dump(args.__dict__,f, indent = 4, ensure_ascii = False)
    print("Succesfully saved meta data file to", save_path)


def load_previous_args(sargs):
    args = None
    
    potential_previous_path = os.path.join(sargs.checkpoint_path, 'experiment_args.json')
    if not os.path.exists(potential_previous_path):
        args = sargs 
        print("Failed to load previous args, no 'experiment_args.json' found at checkpoint-path", sargs.checkpoint_path)
    else:
        with open(os.path.join(sargs.checkpoint_path, 'experiment_args.json'), 'r', encoding='utf-8') as f:
            args = json.load(f)

        for key, val in sargs.__dict__.items():
            if key not in args:
                args[key] = val
            
        args = Namespace(**args)
        args.checkpoint_path = os.path.join(sargs.checkpoint_path, 'best_model.pth')
        args.train = sargs.train 
        args.eval = sargs.eval
        args.isic = sargs.isic
    
    return args


def main(args, return_trainer = False):
    # --- CONFIG ---
    

    BASE_DIR = 'hyperbolic/meta-data'
    CSV_PATH = os.path.join(BASE_DIR, 'all-lesion-diagnoses-meta-data.csv') # From create_diagnostic_tree.py

    if args.outdir:
         
        if args.outdir.startswith('/scratch'):
            OUTPUT_DIR = args.outdir
        else:
            OUTPUT_DIR = os.path.join(
                os.path.dirname(BASE_DIR),
                'hyperbolic' if args.hyperbolic else 'euclidian',
                args.outdir
                )
    else:
        OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR),
                                  'hyperbolic' if args.hyperbolic else 'euclidian',
                                  'tmp_experiment'
                                  )
    
    print("Generating prototypes and saving to ", OUTPUT_DIR)
    PROTOTYPE_PATH,TREE_JSON_PATH = generate_prototypes(args.embed_dim, target_radius=args.radius, outdir = OUTPUT_DIR,
                                                        angular_allocation_metric = args.angular_alloc)

    print(f"Training {'Hyperbolic' if args.hyperbolic else 'Euclidian'} model")
    print("Saving to", OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    train_set = SkinLesionDataset(CSV_PATH, PROTOTYPE_PATH, split = 'train', taxonomy_path=TREE_JSON_PATH)
    val_set = SkinLesionDataset(CSV_PATH, PROTOTYPE_PATH, split = 'val', taxonomy_path=TREE_JSON_PATH)    

    dataset_sanity_check(train_set, val_set)
    
    if args.diagnostic_levels == 1:
        label_mapping = generate_l1_mapping_from_dataset_and_file(train_set, TREE_JSON_PATH)
        train_set.set_diagnostic_mapping(label_mapping)
        val_set.set_diagnostic_mapping(label_mapping)


    weights = None
    args.loss_weight = -1 
    if args.loss_weight != -1:
        weights = get_class_weights(train_set, args.loss_weight)
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.c, pin_memory=True)
    else:
        if args.sampler == 'frequency':
            train_sampler = generate_sampler(train_set, balance_factor = args.sampling_weight / 100, target_level='leaf')
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.c, sampler=train_sampler, pin_memory=True)
        elif args.sampler == 'cascading':
            train_sampler = generate_cascading_sampler(train_set)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.c, sampler=train_sampler, pin_memory=True)
        else:
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.c, pin_memory=True)
        
    
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # --- 2. GRAPH PREP (FOR HED METRIC) ---
    # Load the JSON tree into a NetworkX graph for distance calculations
    with open(TREE_JSON_PATH, 'r') as f:
        tree_dict = json.load(f)
        
    G = nx.Graph()
    # Recursive helper to build graph edges
    def build_graph(node, parent_name=None):
        for name, children in node.items():
            name = name.lower().strip() # Align with dataset keys
            if parent_name:
                G.add_edge(parent_name, name)
            build_graph(children, name)
            
    # Assuming tree_dict is wrapped like {"skin lesion": {...}}
    build_graph(tree_dict)

    # --- 3. MODEL & TRAINER ---
    if args.hyperbolic:
        model = HyperbolicConvNeXt(embedding_dim=EMBEDDING_DIM, backbone_name='convnext_tiny')
    else:
        model = EuclideanConvNeXt(embedding_dim=EMBEDDING_DIM, backbone_name='convnext_tiny')
    

    if not args.eval:
        save_meta_data_json(args, OUTPUT_DIR)

    trainer = SkinCancerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hierarchy_graph=G,
        output_dir=OUTPUT_DIR,
        callbacks=None,
        path_to_hierachy =TREE_JSON_PATH,
        hyperbolic = args.hyperbolic,
        weights = weights, 
        radius_penalty_type = args.radius_penalty
    )
    
    if return_trainer:
        return trainer
    # --- 4. START ---
    if not args.eval:
        trainer.train(epochs=EPOCHS)
    else:
        trainer.load_weights(args.checkpoint_path)
        if args.isic:
            trainer.evaluate_isic(os.path.join(OUTPUT_DIR, 'isic_eval'))
        else:
            trainer.evaluate_and_save_to(OUTPUT_DIR)

def run_sampling_weights(args):
    import copy 
    sampling_weights = [92, 100]

    for weight in sampling_weights:
        new_args = copy.deepcopy(args)
        new_args.sampling_weight = weight
        new_args.outdir = f'{args.exp_name}/sampling_weight_{weight}'
        main(new_args) 


def run_radius_ramping(args, run = 0):
    import copy

    radii = [3.5, 4, 4.5, 5.0, 5.5]

    for i, radius in enumerate(radii):
        if i % 2  == run:
            new_args = copy.deepcopy(args)
            new_args.radius = radius
            new_args.outdir = f"{args.exp_name}/radius_{radius}"
            main(new_args)

def evaluate(args):
    new_args = load_previous_args(args)
    main(new_args)    
    
    
def run_radius_and_weighting(args, run = 1):
    import copy

    radii = [4, 4.5, 5, 5.5]
    weightings = [55, 65, 75, 85]
    
    for i, radius in enumerate(radii):
        for j, weighting in enumerate(weightings):
            if i % 2  == run:
                print("Training and saving for the following")
                print("Radius: ", radius)
                print("Weighting:", weighting)

    for i, radius in enumerate(radii):
        for j, weighting in enumerate(weightings):
            if i % 2  == run:
                new_args = copy.deepcopy(args)
                new_args.radius = radius
                new_args.sampling_weight = weighting
                new_args.outdir = f"{new_args.outdir}/radius_new_{radius}/weighting_{weighting}"
                main(new_args)


def experiment_factory(args):

    experiment_factory_dict = {
        '': main,
        'radius_ramping': run_radius_ramping,
        'sampling_weight_ramping': run_sampling_weights,
        'eval': evaluate,
        'radius_and_weight': run_radius_and_weighting
    }
    return experiment_factory_dict[args.exp_type]


if __name__ == "__main__":

    parser = ArgumentParser(description='Train hyperbolic or regular model')
    parser.add_argument('--hyperbolic', default='hyp', type = str, required=False)
    parser.add_argument('--outdir', default = "", type = str, required=False)
    parser.add_argument('--sampling_weight', default = 70, type = int, required = False)
    parser.add_argument('--embed_dim', default=10, type = int, required=False)
    parser.add_argument('--radius', default=4.0, type = float, required=False)
    parser.add_argument('--exp_name', default = "", type = str, required = False)
    parser.add_argument('--exp_type', default ="", type = str, required = False)
    parser.add_argument('--checkpoint-path', default="", type = str, required=False)
    parser.add_argument('--diagnostic-levels', type = int, default=-1, required=False)
    parser.add_argument('--c', type = int, default = 8, required=False) # Number of workers for the dataloading
    parser.add_argument('--radius_penalty', type = str, default='exponential', required = False) # Type of radius penalty to apply
    parser.add_argument('--angular_alloc', type = str, default='log', required=False)
    parser.add_argument('--run', default=1, type = int, required=False)
    parser.add_argument('--sampler', type = str, default = 'cascading', required = False)
    parser.add_argument('--isic', type=int, default=0)
    args = parser.parse_args()
    # main(args)
    
    hyperbolic_dict = {
        'hyp': True, 
        'euclid': False,
        'mixed': 'mixed',
    }
    args.train = args.exp_type != 'eval'
    args.eval = args.exp_type == 'eval'
    args.hyperbolic = hyperbolic_dict[args.hyperbolic]
    experiment_factory(args)(args)

    # run_radius_ramping(args, args.run)




    
