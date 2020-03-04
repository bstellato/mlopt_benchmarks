import numpy as np
import pandas as pd
import mlopt.settings as stg
import os
import mlopt

def add_details(df, **kwargs):
    """
    Add details to dataframe
    """

    if type(df) == pd.Series:
        for key, val in kwargs.items():
            df[key] = val
    else:
        for key, val in kwargs.items():
            df[key] = [val] * len(df)


def benchmark(m,  # Optimizer
              data_file,
              theta_bar,
              sample_fn,
              dims,
              trees=True
              ):
    """
    Perform benchmark

    Parameters
    ----------
    m : Optimizer
        Problem optimizer.
    data_file : string
        Name of the data file.
    theta_bar : array or dict
        Average value of optimizer.
    sample_fn : Function
        Sampling function.
    dims : dict
        Problem dimensions.
    trees : bool, optional
        Whether or not to train the trees. Defaults to true.
    """

    # Reset random seed
    np.random.seed(1)

    # Get test elements
    theta_test = sample_fn(100)

    data_file_general = data_file + "_general.csv"
    data_file_detail = data_file + "_detail.csv"

    # Loading data points
    already_run = os.path.isfile(data_file_general) and \
        os.path.isfile(data_file_detail)
    if already_run:
        stg.logger.info("Loading data %s" % data_file)
        general = pd.read_csv(data_file_general)
        detail = pd.read_csv(data_file_detail)
    else:

        general = pd.DataFrame()
        detail = pd.DataFrame()

        stg.logger.info("Perform training for %s" % data_file)

        stg.logger.info("Training NN")
        stg.logger.info("-----------\n")

        # Train neural network
        m.train(sampling_fn=sample_fn,
                parallel=True,
                learner=mlopt.PYTORCH)

        nn_general, nn_detail = m.performance(theta_test, parallel=True)
        m.save(data_file + "_nn", delete_existing=True)

        add_details(nn_general, predictor="NN", **dims)
        add_details(nn_detail, predictor="NN", **dims)

        general = general.append(nn_general, ignore_index=True)
        detail = detail.append(nn_detail)


        #  Train and test using optimal trees
        if trees:

            stg.logger.info("Training OCT")
            stg.logger.info("-----------\n")

            # OCT
            m.train(
                    parallel=True,
                    learner=mlopt.OPTIMAL_TREE,
                    hyperplanes=False,
                    save_svg=True)
            oct_general, oct_detail = m.performance(theta_test, parallel=True)
            m.save(data_file + "_oct", delete_existing=True)
            add_details(oct_general, predictor="OCT", **dims)
            add_details(oct_detail, predictor="OCT", **dims)

            #  Combine and store partial results
            general = general.append(oct_general, ignore_index=True)
            detail = detail.append(oct_detail)

            stg.logger.info("Training OCT-H")
            stg.logger.info("-----------\n")

            # OCT-H
            m.train(
                    parallel=True,
                    learner=mlopt.OPTIMAL_TREE,
                    hyperplanes=True,
                    save_svg=True)
            octh_general, octh_detail = m.performance(theta_test,
                                                      parallel=True)
            m.save(data_file + "_octh", delete_existing=True)
            add_details(octh_general, predictor="OCT-H", **dims)
            add_details(octh_detail, predictor="OCT-H", **dims)

            #  Combine and store partial results
            general = general.append(octh_general, ignore_index=True)
            detail = detail.append(octh_detail)

        # Store to csv
        general.to_csv(data_file_general, index=False)
        detail.to_csv(data_file_detail, index=False)

    return general, detail

