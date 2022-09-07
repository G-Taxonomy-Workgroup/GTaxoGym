import argparse
import datetime
import json
import logging
import os

# keywords to be excluded from the final gathered results
EXCLUDE_LIST = ['eta', 'loss', 'lr', 'params']


def parse_args():
    """Parse arguments from command line.
    """
    parser = argparse.ArgumentParser(
        description='Aggregate all results (datasets x perturbations)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--result_dir', default='results/node',
                        help='Result directory')
    parser.add_argument('--config_name', default='node_default',
                        help='Configuration name')
    parser.add_argument('--output_dir', default='agg_results',
                        help='Output directory')
    parser.add_argument('--output_fn', default='node_results.json',
                        help='Output filename')

    return parser.parse_args()


def init_logger(args):
    """Initialize logger.

    Setup result output directory and log output file. Add stream handler to
    log results to stdout.

    Return:
        Log file name

    """
    # check whether output directory exists and create if not
    log_dir = os.path.join(args.output_dir, 'logs')
    for directory in [args.output_dir, log_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # setup logging file name based on date and log to stdout also
    log_fn = f'{datetime.date.today()}.log'
    log_fp = os.path.join(log_dir, log_fn)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fp, mode='w'),
            logging.StreamHandler(),
        ]
    )

    # log parsed arguments
    logging.info(args)

    return log_fp


def iterate_over_settings(args):
    """Iterate over the result directories.

    Iterate over the combinations of datasets x perturbations and yield the
    corresponding result directory, which contains three sub directories:
    ``train``, ``val``, and ``test``. The target file is ``best.json``, which
    contains one line json about the aggregated evaluation of the dataset
    across all runs (different seeds).

    The result directory structure looks something like:
    <result_dir>/<pert_type>/<dataset_name>/<config_name>/agg

    """
    for pert_type in [x for x in os.listdir(args.result_dir) if os.path.isdir(os.path.join(args.result_dir, x))]:
        pert_dir = os.path.join(args.result_dir, pert_type)
        for dataset_name in [x for x in os.listdir(pert_dir) if os.path.isdir(os.path.join(pert_dir, x))]:
            agg_dir = os.path.join(pert_dir, dataset_name, args.config_name, 'agg')
            yield dataset_name, pert_type, agg_dir


def add_result(results, scores, dataset_name, pert_type, split):
    """Filter and add result to the result list.

    Given the dictionary ``scores`` of aggregated evaluations, exclude the
    standard deviation values and also the exclude keywords, then append the
    filtered results along with the information of the experiment to ``results``

    Args:
        results (list): list of final aggregated evaluations
        scores (dict): dictionary of aggregated evaluations for a specific
            dataset
        dataset_name (str): name of the dataset
        pert_type (str): perturbation type
        split (str): ``'train'``, ``'val'``, or ``'test'``

    """
    new_result = {}
    new_result['Dataset'] = dataset_name
    new_result['Perturbation'] = pert_type
    new_result['Split'] = split
    for kw, val in scores.items():
        if kw in EXCLUDE_LIST or kw.endswith('_std'):  # also exclude std values
            continue
        new_result[f'score-{kw}'] = val
    results.append(new_result)


def _print_elements(name, elements):
    logging.info(f'Total number of {name}: {len(elements)}')
    for element in elements:
        logging.info(f'    {element}')


def check_completeness(datasets, perturbations, results):
    """Check if all combination of datasets and perturbations are evaluated.
    """
    _print_elements('datasets', datasets)
    _print_elements('perturbations', perturbations)
    dataset_perturbation_pairs = [
        (result['Dataset'], result['Perturbation']) for result in results
    ]

    miss_count = 0
    for dataset in datasets:
        for perturbation in perturbations:
            if (dataset, perturbation) not in dataset_perturbation_pairs:
                logging.warning(f'Missing result: {dataset} x {perturbation}')
                miss_count += 1

    if miss_count > 0:
        tot_count = len(datasets) * len(perturbations)
        logging.warning(f'{miss_count} of {tot_count} experiments missing!')
    else:
        logging.info(f'Completed experiments!')


def main():
    """Main function for result aggregation script.
    """
    args = parse_args()
    log_fp = init_logger(args)

    datasets = set()
    perturbations = set()

    results = []
    for dataset_name, pert_type, agg_dir in iterate_over_settings(args):
        datasets.add(dataset_name)
        perturbations.add(pert_type)
        logging.info(f'Loading results from: {agg_dir}')

        for split in ['train', 'val', 'test']:
            best_agg_fp = os.path.join(agg_dir, split, 'best.json')
            if not os.path.isfile(best_agg_fp):
                logging.warning(f'File does not exist: {best_agg_fp!r}')
                continue

            try:
                with open(best_agg_fp, 'r') as f:
                    scores = json.load(f)
                    add_result(results, scores, dataset_name, pert_type, split)
            except:
                logging.warning(f'Could not print: {str(best_agg_fp)}')

    check_completeness(datasets, perturbations, results)

    with open(f'{args.output_dir}/{args.output_fn}', 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Finished gathering results, log file saved to: {log_fp}')


if __name__ == "__main__":
    main()
