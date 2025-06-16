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
        for row in seq_count_index.itertuples(index=False):
            file.write(f">{row.index}_{row.count}\n{row.seq}\n")
    return output_fasta

def swarm_func(input_fasta: str, 
               swarm_args: str, 
               n_cpus=1):
    """
    This function executes shell commands for swarm clustering.
    """
    # Prepare output file paths
    output_cluster = input_fasta.replace('.fasta', '.swarms')
    output_stat = input_fasta.replace('.fasta', '_stat.csv')
    # Prepare the command arguments as a list for safety
    command = ['swarm', input_fasta] + swarm_args.split() + ['-t', str(n_cpus), '-o', output_cluster, '-s', output_stat]
        
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

def cluster_withincluster (cluster_file, 
                           neighbourhood_csv):

    cluster_corrected = cluster_file.replace('.txt', '_corrected.txt')
    with open(cluster_corrected, 'w') as file:
        file.write('')

    modularity_score_file = cluster_file.replace('.txt', '_moduscore.txt')
    with open(modularity_score_file, 'w') as file:
        file.write('')

    neighbours_df = pd.read_csv(neighbourhood_csv)
    df_columns = neighbours_df.columns.tolist()

    with open(cluster_file, 'r') as lines:
        for line in lines:
            headers = line.strip('\n').split(' ')
            if len(headers) < 2:    # Continue to next line if <2 members in this cluster.
                continue
            
            ind_list = [int(header.split('_')[0]) for header in headers]   # a list of int
            ind_header_dict ={int((header.split('_')[0])):header for header in headers}
            df = neighbours_df[neighbours_df[df_columns[0]].isin(ind_list)]
            B = nx.from_pandas_edgelist(
                df, 
                source=df_columns[0], 
                target=df_columns[1], 
                edge_attr=df_columns[2])
            points = set(df[df_columns[0]])
            projected_graph = bipartite.weighted_projected_graph(B, points)
            edges_with_weights = [(edge[0], edge[1], edge[2]['weight']) for edge in projected_graph.edges(data=True)]
            ig_graph = ig.Graph.TupleList(
                edges_with_weights, 
                edge_attrs=['weight'], 
                vertex_name_attr='name', 
                directed=False)
            ig_graph.vs['name'] = list(projected_graph.nodes())
            partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
            
            # Record the results
            clusters = [[ig_graph.vs["name"][v] for v in cluster] for cluster in partition]
            modularity_score = partition.modularity
            with open(modularity_score_file, 'a') as file:
                file.write(f">{headers[0]}\nModularity Score: {modularity_score}\nClusters: {clusters}\n")
            if not isinstance(modularity_score, float):
                continue

            # Record corrected clusters
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                cluster = list(map(int, cluster))
                cluster_headers = [ind_header_dict[ind] for ind in cluster]
                cluster_headers = sorted(cluster_headers, key=lambda header: int(header.split('_')[1]), reverse=True)
                with open(cluster_corrected, 'a') as file:
                    file.write(f"{' '.join(cluster_headers)}\n")

    return cluster_corrected, modularity_score_file

def cluster_uxis_seqs(input_csv: str,
                      input_seqindexmap_list: list,
                      swarm_args='-d 1 -f -n',
                      uxis_to_process=['umi1', 'umi2'],
                      
                      correct_cluster=True,
                      neighbours_dict={},
                      correct_n_chunks=1,
                      correct_temp_remove=True,
                      n_cpus=1):
    
    output_maindir = os.path.dirname(input_csv)
    reindexed_csv = input_csv.replace('.csv', f"_{''.join(uxis_to_process)}reindexed.csv")

    for uxi in uxis_to_process:
        uxi_ind = uxi + '_ind'
        # Inital clustering
        logging.info(f"Clustering {uxi} on {input_csv}")
        logging.info(f"Clustering {uxi}: Generating FASTA file...")
        initial_fasta = generate_fasta(input_seqindexmap_list,
                                       output_maindir,
                                       uxi)
        logging.info(f"Clustering {uxi}: Generated {initial_fasta}")
        logging.info(f"Clustering {uxi}: Swarm clustering...")
        cluster_file, _ = swarm_func(initial_fasta, swarm_args, n_cpus)
        logging.info(f"Clustering {uxi}: Initial clustering completed.")
        
        # Correct the uxi clusters
        if correct_cluster and neighbours_dict:
            # Set temp dir
            output_dir = os.path.join(os.path.dirname(input_csv), f"{uxi}_temp")
            os.makedirs(output_dir, exist_ok=True)

            # Acquire the neighbourhood
            logging.info(f"Correcting {uxi} clusters: Acquiring {uxi} neighbourhood.")
            neighbour_uxiindtype = neighbours_dict[uxi_ind]
            df = pd.read_csv(input_csv)
            df = df[[uxi_ind, neighbour_uxiindtype, 'n_reads']]
            df = df.groupby([uxi_ind, neighbour_uxiindtype])['n_reads'].sum().reset_index()
            logging.info(f"Correcting {uxi} clusters: neighbourhood df (edgelist) from {input_csv}\n{df.head()}")

            logging.info(f"Correcting {uxi} clusters: Splitting clusters.")
            with open(cluster_file, 'r') as file:
                cluster_lines = file.readlines()
            np.random.seed(42)
            np.random.shuffle(cluster_lines)

            base_chunk_size = len(cluster_lines) // correct_n_chunks
            extra_lines = len(cluster_lines) % correct_n_chunks

            start = 0
            temp_chunks = list()
            temp_neighbourhooddf = list()
            for i in range(correct_n_chunks):
                end = start + base_chunk_size + (1 if i < extra_lines else 0)
                # Split the clusters
                chunk = cluster_lines[start:end]
                chunk_filename = f"{output_dir}/{uxi}_chunk_{i}.txt"
                with open(chunk_filename, 'w') as chunk_file:
                    chunk_file.writelines(chunk)
                temp_chunks.append(chunk_filename)
                # Split the neighbourhood df
                ind_list = [line.strip().split(' ') for line in chunk]
                ind_list = [item for sublist in ind_list for item in sublist]
                ind_list = [int(header.split('_')[0]) for header in ind_list]   # a list of int
                small_df = df[df[uxi_ind].isin(ind_list)]
                small_df_filename = f"{output_dir}/{uxi}_neighbourhood_{i}.csv"
                small_df.to_csv(small_df_filename, index=False)
                temp_neighbourhooddf.append(small_df_filename)
                start = end

            del df, cluster_lines
            gc.collect

            # Multi-processing
            corrected_files = list()
            moduscore_files = list()
            logging.info(f"Correcting {uxi} clusters: Multi-processing...")
            futures = []
            with ProcessPoolExecutor(max_workers=min(correct_n_chunks, n_cpus)) as executor:
                for small_cluster, small_neighbourhooddf in zip(temp_chunks, temp_neighbourhooddf):
                    future = executor.submit(cluster_withincluster, small_cluster, small_neighbourhooddf)
                    futures.append(future)
            for future in futures:
                result = future.result()
                small_cluster_corrected, moduscore_file = result  # Unpack results
                corrected_files.append(small_cluster_corrected)
                moduscore_files.append(moduscore_file)

            corrected_swarm = cluster_file.replace('.swarms', '_corrected.swarms')
            with open(corrected_swarm, 'w') as outfile:
                for file_path in corrected_files:
                    with open(file_path, 'r') as infile:
                        content = infile.read()
                        outfile.write(f"{content}\n")
            logging.info(f"Correcting {uxi} clusters: Corrected clusters saved to {corrected_swarm}")

            moduscore_merged = cluster_file.replace('.swarms', '_moduscore_merged.txt')
            with open(moduscore_merged, 'w') as outfile:
                for file_path in moduscore_files:
                    with open(file_path, 'r') as infile:
                        content = infile.read()
                        outfile.write(f"{content}\n")
            logging.info(f"Correcting {uxi} clusters: Modularity score saved to {moduscore_merged}")

            # Clean up temporary files
            if correct_temp_remove:
                logging.info(f"Correcting {uxi} clusters: Removing temp files")
                for file_list in [temp_chunks, temp_neighbourhooddf, corrected_files, moduscore_files]:
                    for file_path in file_list:
                        os.remove(file_path)
    
    # Build index-reindex dictionary
    uxi_reindex_dict = dict()
    for uxi in uxis_to_process:
        logging.info(f"Reindexing {uxi}...")
        uxi_reindex_dict[uxi] = dict()
        
        swarm_files = glob.glob(os.path.join(output_maindir, '**', f'{uxi}_corrected.swarms'), recursive=True)
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
                    for indx in uxi_ind_list:
                        uxi_reindex_dict[uxi][indx] = uxi_ind_list[0]

    # Reindex
    uxis_df = pd.read_csv(input_csv)
    for uxi in uxis_to_process:
        uxi_ind = f"{uxi}_ind"
        uxi_reind = f"{uxi}_reind"
        uxis_df[uxi_reind] = uxis_df[uxi_ind].copy()
        uxis_df[uxi_reind] = uxis_df[uxi_reind].map(lambda x: uxi_reindex_dict[uxi].get(x, x))
        logging.info(f"Reindexing {uxi}: {uxis_df[uxi_ind].nunique()} -> {uxis_df[uxi_reind].nunique()}")
    uxis_df.to_csv(reindexed_csv, index=False)
    logging.info(f"Reindexed edges saved to: {reindexed_csv}")

    return reindexed_csv
