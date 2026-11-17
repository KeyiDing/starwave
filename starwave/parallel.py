import torch
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm  # Import tqdm for the progress bar
import numpy as np
from getmags import *
import extinction


def custom_simulate_for_sbi(
    simulator, proposal, num_simulations, num_workers=1, simulation_batch_size=1
):
    theta = proposal.sample((num_simulations,))
    batch_size = simulation_batch_size if simulation_batch_size else num_simulations
    theta_batches = torch.split(theta, batch_size)

    def simulator_wrapper(batch):
        torch.set_num_threads(2)  # Force exactly 2 threads per worker dynamically
        return simulator(batch)

    with Pool(nodes=num_workers) as pool:
        # 1. Use imap() instead of map() to get a live iterator
        results_iterator = pool.imap(simulator_wrapper, theta_batches)
        
        # 2. Wrap the iterator in tqdm, specifying the total number of batches
        raw_results = list(
            tqdm(
                results_iterator, 
                total=len(theta_batches), 
                desc=f"Simulating ({num_workers} workers)"
            )
        )

    x_results = [torch.as_tensor(res, dtype=torch.float32) for res in raw_results]
    x = torch.cat(x_results, dim=0)

    return theta, x

def _process_star(args):
    (ii, mass, binq, age, feh, dm, av, 
     lim_logmass, iso_int, bands, band_lambdas, Rv) = args

    if mass < lim_logmass or np.isnan(age) or np.isnan(feh):
        return ii, None, None

    input_mag = get_absolute_mags(mass, age, feh, binq, iso_int, bands)
    calc_mag = input_mag + dm
    
    # Note: See optimization tip below regarding this extinction call
    calc_ext = np.array([extinction.ccm89(np.array([l]), av, Rv)[0] for l in band_lambdas])

    return ii, calc_mag, calc_ext

