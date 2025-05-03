from CollectUtils import *

np.random.seed(42)
file_ex = "Data/BurgersExact.txt"
exact_solution = np.loadtxt(file_ex)

base_path_list = ["RarefactionWave"]

for base_path in base_path_list:
    print("#################################################")
    print(base_path)

    b = False
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame()
    selection_criterion = "metric"
    eval_metric = "rel_L2_norm"
    threshold = 0.005
    plot_color = "reset-freq"
    mode = "min"
    mode_ret = "mean"

    Nu_list = []
    Nf_list = []

    L2_norm = []
    criterion = []
    best_retrain_list = []
    list_models_setup = list()

    for subdirec in directories_model:
        print(subdirec)
        model_path = base_path

        sample_path = model_path + "/" + subdirec
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        retr_to_check_file = None
        for ret in retrainings_fold:
            print(sample_path + "/" + ret + "/EnsembleInfo.csv")
            if os.path.isfile(sample_path + "/" + ret + "/EnsembleInfo.csv"):
                retr_to_check_file = ret
                break

        setup_num = int(subdirec.split("_")[1])
        if retr_to_check_file:
            model_info = pd.read_csv(os.path.join(sample_path, retr_to_check_file, "EnsembleInfo.csv"), 
                               header=None, sep=",", index_col=0)
            model_info = model_info.transpose()
            model_info = model_info.reset_index().drop(columns=["index"])
            
            best_training = select_over_retrainings(
                sample_path,
                selection=selection_criterion, 
                mode=mode_ret,
                exact_solution=exact_solution
            )
            training_data = best_training.to_frame().transpose()
            training_data = training_data.reset_index()
            training_data = training_data.drop(columns=["index"])
            
            combined_data = pd.concat([model_info, training_data], axis=1)
            combined_data["setup"] = setup_num
            sensitivity_df = pd.concat([sensitivity_df, combined_data], ignore_index=True)
        else:
            print(f"File not found: {sample_path}/Information.csv")

    ordered_results = sensitivity_df.sort_values(by=selection_criterion)
    
    if mode.lower() == "min":
        chosen_setup = ordered_results.iloc[0]
    elif mode.lower() == "max":
        chosen_setup = ordered_results.iloc[-1]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
        
    print(f"Best setup number: {chosen_setup['setup']}")
    print(chosen_setup)
    
    output_file = f"{base_path}/best.csv"
    chosen_setup.to_csv(output_file, header=False, index=True)

    column_mapping = {'reset_freq': 'reset-freq'}
    ordered_results = ordered_results.rename(columns=column_mapping)

    figure = plt.figure()
    plt.grid(which="both", linestyle=":", alpha=0.7)
    sns.scatterplot(
        data=ordered_results, 
        x=selection_criterion, 
        y=eval_metric, 
        hue=plot_color
    )

    plt.xlabel(r'$\varepsilon_T$')
    plt.ylabel(r'$\varepsilon$')
    plt.xscale("log")
    plt.yscale("log")
    
    img_filename = f"{base_path}/et_vs_eg_{selection_criterion}_{mode_ret}.png"
    plt.savefig(img_filename, dpi=400)
