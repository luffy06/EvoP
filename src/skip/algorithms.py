import logging
import numpy as np

from utils.genetic_algorithm import GA
from utils.eval_utils import compute_loss_ga

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_in_genetic_algorithm(model, nrm_blocks, train_ids, algo_configs={}):
    func_kwargs = {
        'model': model, 
        'train_ids': train_ids,
        'nrm_blocks': nrm_blocks
    }
    ga = GA(
        func=compute_loss_ga, 
        func_kwargs=func_kwargs,
        n_dim=model.num_layers, 
        n_ones=nrm_blocks,
        lb=0,
        ub=1,
        precision=1,
        **algo_configs
    )
    logger.info(f"GA initialized.")
    skip_list, best_loss = ga.run()
    
    skip_list = [i for i in range(model.num_layers) if skip_list[i] == 1]
    return best_loss, skip_list

def run_in_beam_search(model, nrm_blocks, train_ids, num_beam=1):
    beam_skip_layers = [[]]
    best_loss =[np.inf for _ in range(num_beam)]
    for round_i in range(nrm_blocks):
        beam_loss = []
        for beam_i in range(len(beam_skip_layers)):
            logger.info(f"Round {round_i + 1}/{nrm_blocks}, Beam {beam_i + 1}/{len(beam_skip_layers)}")
            min_loss = []
            for layer_idx in range(model.num_layers):
                if layer_idx in beam_skip_layers[beam_i]:
                    continue
                local_skip_layers = beam_skip_layers[beam_i] + [layer_idx]
                sum_loss = compute_loss(local_skip_layers, model, train_ids, nrm_blocks)                
                logger.info(f"Beam {beam_i} skip layers {local_skip_layers} loss: {sum_loss}")
                min_loss.append((sum_loss, beam_i, layer_idx))
            min_loss = sorted(min_loss, key=lambda x: x[0])
            min_loss = min_loss[:num_beam]
            beam_loss.extend(min_loss)
        beam_loss = sorted(beam_loss, key=lambda x: x[0])
        logger.info(f"Current skip layers: {beam_skip_layers}")
        logger.info(f"Beam loss: {beam_loss}")
        beam_loss = beam_loss[:num_beam]
        skip_layers = []
        for i, loss_i in enumerate(beam_loss):
            loss, beam_i, skip_layer_idx = loss_i
            skip_layers.append(beam_skip_layers[beam_i] + [skip_layer_idx])
            best_loss[i] = loss
        beam_skip_layers = skip_layers
    best_loss = best_loss[0]
    skip_layers = beam_skip_layers[0]
    return best_loss, skip_layers
