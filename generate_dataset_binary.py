import argparse
from data_loader.dataset import DatasetBase

def main(args):
    binary_dataset = DatasetBase(directory=args.data_dir, raw=True, \
        corpus_suffix=args.corpus_suffix,
        doc2phrase_suffix=args.doc2phrase_suffix,
        topic_suffix=args.topic_suffix,
        topic_hier_suffix=args.topic_hier_suffix,
        topic_feat_suffix=args.topic_feat_suffix,
        topic_triple_suffix=args.topic_triple_suffix,
        output_suffix=args.output_suffix
    )

if __name__ == '__main__':
    """
    python generate_dataset_binary.py --data_dir <DATA_DIR>
    """
    args = argparse.ArgumentParser(description='Generate binary data pickle')
    args.add_argument('--data_dir', required=True, type=str, help='path to data directory')
    args.add_argument('--corpus_suffix', type=str, default="")
    args.add_argument('--doc2phrase_suffix', type=str, default="")
    args.add_argument('--topic_suffix', type=str, default="")
    args.add_argument('--topic_hier_suffix', type=str, default="")
    args.add_argument('--topic_feat_suffix', type=str, default="")
    args.add_argument('--topic_triple_suffix', type=str, default="")
    args.add_argument('--output_suffix', type=str, default="")
    args = args.parse_args()
    main(args)
