import pandas as pd
import os
import subprocess
import logging
import SeqIO
import numpy as np
import glob

def generate_fasta(input_seqindexmap_list: list,
                   output_maindir: str,
                   uxi='umi1'):
    """
    This function generates fasta files for each uxi.
    """ 
    seq_count_index_filename = str()
    for filename in input_seqindexmap_list:
        if filename.endswith(f"{uxi}.csv"):
            seq_count_index_filename = filename
            break
    
    seq_count_index = pd.read_csv(seq_count_index_filename)
    fasta_dir = f"{output_maindir}/{uxi}_clustering"
    os.makedirs(fasta_dir, exist_ok=True)
        
    output_fasta = f"{fasta_dir}/{uxi}.fasta"
    with open(output_fasta, "w") as file:
        for _, row in seq_count_index.iterrows():
            seq = row['seq']
            count = row['count']
            ind = row['index']
            file.write(f">{ind}_{count}\n{seq}\n")

    return output_fasta


def swarm_func(input_fasta: str, 
               swarm_args: str):
    """
    This function executes shell commands for swarm clustering.
    """
    # Prepare output file paths
    output_cluster = input_fasta.replace('.fasta', '.swarms')
    output_stat = input_fasta.replace('.fasta', '_stat.csv')
    # Prepare the command arguments as a list for safety
    command = ['swarm', input_fasta] + swarm_args.split() + ['-o', output_cluster, '-s', output_stat]
        
    try:
        # Run the command using subprocess
        result = subprocess.run(command, capture_output=True, text=True)        
        # Check the result of the command
        if result.returncode != 0:
            logging.error(f"Swarm clustering failed for: {input_fasta}")
            logging.error(f"Error message: {result.stderr}")
        
    except Exception as e:
        logging.error(f"Exception occurred while running swarm for {input_fasta}: {str(e)}")
    return output_cluster, output_stat


def filterby_commonneighbours(neighbours_cache: dict,
                              fasta_file: str,
                              cluster_file: str,
                              cnr_file: str,

                              fasta_filteredout: str,
                              cluster_updated: str,

                              commonneighbours_threshold=0.75):
    
    """
    Filters clusters based on common neighbors and generates a new .fasta file with filtered-out sequences.
    
    Parameters:
    -----------
    input_csv : dict
        The nieghbourhood dictionary
    fasta_file : str
        Path to the initial .fasta file containing the sequences to be filtered. 
    cluster_file : str
        Path to the file containing clustering information, where each line represents a cluster with sequence IDs and counts.
    cnr_file: str
        Path to the output common neighbourhood ratio.
    fasta_filteredout : str
        Path to the output .fasta file where filtered-out sequences will be saved.
    cluster_updated : str
        Path to the output file where updated cluster information will be saved, removing clusters based on the filtering criteria.
    commonneighbours_threshold : float
        Threshold ratio for common neighbors used to determine if a sequence should be filtered out. Default is 0.75.
    
    Returns:
    --------
    str
        The path to the .fasta file with filtered-out sequences.

    Process:
    --------
    1. Reads input CSV to construct a neighbors cache, grouping sequences by their neighbors for the specified identifier type.
    2. Reads the cluster file line-by-line and compares each sequence's neighbors with those of the most frequent sequence in the cluster.
    3. Calculates the ratio of common neighbors and filters out sequences that don't meet the threshold.
    4. Writes the updated cluster information to the specified output file and generates a .fasta file of sequences removed from clusters.
    
    Example:
    --------
    >>> filterby_commonneighbours("data.csv", "initial.fasta", "clusters.swarms",
                                  "filtered_out.fasta", "updated_clusters.swarms",
                                  "umi1_ind", commonneighbours_threshold=0.75)
    
    Note:
    -----
    This function assumes the presence of a `SeqIO` module for reading and writing .fasta files.
    Temporary files, if any, should be deleted externally as this function only appends to the specified output files.
    """

    # Check for common neighbours
    filtered_clusters_list = []
    common_neighbours_ratio_dict = {}
    headers_to_remove = set()
    with open(cluster_file, 'r') as lines:
        for line in lines:
            line_inlist = line.strip('\n').split()
            if len(line_inlist) < 2:    # Continue to next line if <2 members in the cluster.
                continue
            
            uxi_ind_list = sorted(line_inlist, key=lambda header: int(header.split('_')[1]), reverse=True)
            uxi_ind_list = [int(header.split('_')[0]) for header in uxi_ind_list]   # a list of int
            ind_header_dict ={int((header.split('_')[0])):header for header in line_inlist}

            neighbours_0 = neighbours_cache[uxi_ind_list[0]] # Get neighbours for the most abundant uxi (index 0)
            common_neighbours_ratio_dict[ind_header_dict[uxi_ind_list[0]]] = []
            for ind in uxi_ind_list[1:]:
                neighbours_i = neighbours_cache[ind]
                common_neighbours_ratio = len(neighbours_0.intersection(neighbours_i)) / len(neighbours_i)
                common_neighbours_ratio_dict[ind_header_dict[uxi_ind_list[0]]].append(common_neighbours_ratio)   # record the common neighbours ratio for large cluster (>1 members)
                if common_neighbours_ratio <= commonneighbours_threshold:
                    headers_to_remove.add(ind_header_dict[ind])
            if np.sum(common_neighbours_ratio_dict[ind_header_dict[uxi_ind_list[0]]]) == 0:
                headers_to_remove.add(ind_header_dict[uxi_ind_list[0]])
            
            # Filter the line and write the updated line to the filtered file
            line_filtered = [item for item in line_inlist if item not in headers_to_remove]
            if line_filtered:
                filtered_clusters_list.append(' '.join(line_filtered))
    with open(cluster_updated, 'a') as file:
        for line in filtered_clusters_list:
            file.write(line+'\n')

    # Record common neighbours ratio
    with open(cnr_file, 'w') as out_file:
         for key, values in common_neighbours_ratio_dict.items():
                values_str = " ".join(f"{value:.4f}" for value in values)
                out_file.write(f"{key}:{values_str}\n")
    logging.info(f"Saved the common membership ratio: {cnr_file}")
                    
    # Generate .fasta file for the filtered out uxis
    with open(fasta_filteredout, "w") as out_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            if record.id in headers_to_remove:
                SeqIO.write(record, out_file, "fasta")
    logging.info(f"Saved the filtered-out sequences: {fasta_filteredout}")

    return common_neighbours_ratio_dict


def cluster_umi1_umi2(input_csv: str,
                      input_seqindexmap_list: list,
                      
                      swarm_args='-d 1 -f -n',
                      
                      n_recluster_setup=3,
                      commonneighbours_threshold=0.75,

                      uxis_to_process=['umi1', 'umi2'],
                      neighbours_dict={'umi1_ind': 'umi2_ind', 'umi2_ind': 'umi1_ind'},
                      
                      check_membership=True):
    
    """
    Clusters unique molecular identifiers (UMIs) in the dataset and correct based on the neighbouring relations 
    and reindexes them.

    Parameters:
    -----------
    input_csv : str
        Path to the input CSV file containing UMI data.     
    input_seqindexmap_list : list
        List containing mappings of sequence indices to be clustered and processed.
    swarm_args : str
        Arguments to be passed to the swarm clustering function. Default is '-d 1 -f -n'.        
    n_recluster_setup : int
        Number of iterations for reclustering to ensure convergence. Default is 3.        
    commonneighbours_threshold : float
        Threshold for filtering based on common neighbors during reclustering. Default is 0.75.        
    uxis_to_process : list
        List of UMI columns to process and reindex. Default is ['umi1', 'umi2'].       
    neighbours_dict : dict
        Dictionary mapping each UMI index column to its neighbor column for comparison. 
        Default is {'umi1_ind': 'umi2_ind', 'umi2_ind': 'umi1_ind'}.      
        
    Returns:
    --------
    str
        Path to the final reindexed CSV file after clustering.
        
    Process:
    --------
    1. Initializes output directory for each UMI, generates initial .fasta files, and performs clustering with `swarm_func`.
    2. Iteratively reclusters based on common neighbors, generating new .fasta files for each iteration.
       - Reclustering stops if no data is left for further clustering in the filtered .fasta files.
    3. Builds a reindexing dictionary by reading and parsing updated clustering files.
    4. Reads the input CSV, applies the reindexing map, and saves the final reindexed DataFrame to a new CSV.
    
    Example:
    --------
    >>> cluster_umi1_umi2("umi_data.csv", ["seq_map1", "seq_map2"])
    
    Note:
    -----
    This function assumes the presence of helper functions such as `generate_fasta`, `swarm_func`, and `filterby_commonneighbours`
    for proper functioning. Intermediate clustering files and temporary .fasta files are deleted to conserve memory.
    """

    output_maindir = os.path.dirname(input_csv)
    os.makedirs(output_maindir, exist_ok=True)
    reindexed_csv = input_csv.replace('.csv', f"_{''.join(uxis_to_process)}reindexed.csv")

    for uxi in uxis_to_process:
        uxi_ind = uxi + '_ind'
        ## Inital clustering
        logging.info(f"Clustering {uxi}...")
        logging.info(f"Clustering {uxi}: Generating .fasta file...")
        initial_fasta = generate_fasta(input_seqindexmap_list,
                                       output_maindir,
                                       uxi)
        logging.info(f"Clustering {uxi}: Swarm clustering...")
        cluster_file, _ = swarm_func(initial_fasta, swarm_args)
        logging.info(f"Clustering {uxi}: Initial clustering completed.")
        ## Correct the clusters based on the members' common neighbours
        cluster_updated = cluster_file.replace('.swarms', '_updated.swarms')
        with open(cluster_updated, 'w') as file:
            file.write('')
        
        recluster_file = cluster_file
        if check_membership:
            # Acquring the neighbourhood dictionary
            logging.info(f"Clustering {uxi}: Acquiring {uxi} neighbourhood dictionary.")
            df = pd.read_csv(input_csv)
            neighbour_uxiindtype = neighbours_dict[uxi_ind]
            neighbours_cache = df.groupby(uxi_ind)[neighbour_uxiindtype].apply(set).to_dict()
            logging.info(f"Clustering {uxi}: Acquired the neighbourhood dictionary.")
            # Examining the membership with the neighbourhood dictionary
            for i in range(n_recluster_setup):
                logging.info(f"Clustering {uxi}: Checking the membership and correcting.")
                cnr_file = initial_fasta.replace('.fasta', f'_{i}_cnr')
                fasta_filteredout = initial_fasta.replace('.fasta', f'_{i+1}.fasta')
                cnr_dict = filterby_commonneighbours(
                    neighbours_cache,
                    initial_fasta,
                    recluster_file,
                    cnr_file,
                    fasta_filteredout,
                    cluster_updated,
                    commonneighbours_threshold)
                # Check if all seqs are clustered
                if os.path.getsize(fasta_filteredout) == 0:
                    logging.info(f"Clustering {uxi}: Converged after round {i} reclustering. All seqs are clustered.")
                    break
                # Check if all clusters' members neighbourhood
                if np.sum([np.sum(values) for values in cnr_dict.values()]) == 0:
                    logging.info(f"Clustering {uxi}: Converged after round {i} reclustering. The residue seqs have no common neighbours.")
                    break
                # Recluster
                logging.info(f"Clustering {uxi}: Round {i + 1} reclustering.")
                recluster_file, _ = swarm_func(fasta_filteredout, swarm_args)
                with open(cluster_updated, 'a') as file:
                    with open(recluster_file, 'r') as lines:
                        for line in lines:
                            file.write(line)
    
    ## Build index-reindex dictionary
    uxi_reindex_dict = dict()
    for uxi in uxis_to_process:
        logging.info(f"Reindexing {uxi}...")
        uxi_reindex_dict[uxi] = dict()
        
        swarm_files = glob.glob(os.path.join(output_maindir, '**', f'{uxi}_updated.swarms'), recursive=True)
        if len(swarm_files) == 0:
            swarm_files = glob.glob(os.path.join(output_maindir, '**', f'{uxi}.swarms'), recursive=True)
        logging.info(f"Reindexing {uxi} with {' '.join(swarm_files)}")
        
        for swarm_file in swarm_files:
            with open(swarm_file, 'r') as lines:
                for line in lines:
                    line_inlist = line.strip().split(' ')
                    # Continue to next line if <2 members in the cluster.
                    if len(line_inlist) < 2:    
                        continue
                    uxi_ind_list = sorted(line_inlist, key=lambda header: int(header.split('_')[1]), reverse=True)
                    uxi_ind_list = [int(header.split('_')[0]) for header in uxi_ind_list]   # a list of int
                    # Reindex all uxi_ind to the most frequent one in the group
                    for uxi_ind in uxi_ind_list:
                        uxi_reindex_dict[uxi][uxi_ind] = uxi_ind_list[0]

    ## Reindex
    uxis_df = pd.read_csv(input_csv)
    for uxi in uxis_to_process:
        uxis_df[f"{uxi}_reind"] = uxis_df[f"{uxi}_ind"].copy()
        uxis_df[f"{uxi}_reind"] = uxis_df[f"{uxi}_reind"].map(lambda x: uxi_reindex_dict[uxi].get(x, x))
    uxis_df.to_csv(reindexed_csv, index=False)
    logging.info(f"Reindexed edges saved to: {reindexed_csv}")

    return reindexed_csv
