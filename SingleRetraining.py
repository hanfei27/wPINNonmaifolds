import os
import sys
import json
import random
import subprocess

random.seed(42)

print("Start Retrainings")

sampling_seed = int(sys.argv[1])
n_coll = int(sys.argv[2])
n_u = int(sys.argv[3])
n_int = int(sys.argv[4])
folder_path = sys.argv[5]
validation_size = float(sys.argv[6])
network_properties = json.loads(sys.argv[7])
shuffle = sys.argv[8]
cluster = sys.argv[9]
GPU = str(sys.argv[10])
n_retrain = int(sys.argv[11])

seeds = list()
for i in range(n_retrain):
    seeds.append(random.randint(1, 500))
print(seeds)
import numpy as np
seeds = np.array(seeds)
seeds = (seeds - min(seeds))*49/(max(seeds) -min(seeds))
seeds = np.round(seeds).astype(int)

os.mkdir(folder_path)

for retrain in range(len(seeds)):
    folder_path_retraining = folder_path + "/Retrain_" + str(retrain)
    command_args = []
    command_args.append(str(sampling_seed))
    command_args.append(str(n_coll))
    command_args.append(str(n_u))
    command_args.append(str(n_int))
    command_args.append(str(folder_path_retraining))
    command_args.append(str(validation_size))
    
    # Format network properties based on platform
    is_unix_like = sys.platform in ["linux", "linux2", "darwin"]
    if is_unix_like:
        formatted_properties = "\'" + str(network_properties).replace("\'", "\"") + "\'"
    else:
        formatted_properties = str(network_properties).replace("\'", "\"")
    
    command_args.append(formatted_properties)
    command_args.append(str(seeds[retrain]))
    command_args.append(shuffle)

    # Execute based on platform and environment
    if is_unix_like:
        if cluster == "true":
            if GPU != "None":
                execution_cmd = f"bsub -W 12:00 -R \"rusage[mem=16384,ngpus_excl_p=1]\" -R \"select[gpu_model0=={GPU}]\" python3 wPINNS.py "
                print(execution_cmd)
            else:
                execution_cmd = "bsub -W 12:00 -R \"rusage[mem=8192]\" python3 wPINNS.py "
        else:
            execution_cmd = "python3 wPINNS.py "
            
        # Append all arguments to command
        for param in command_args:
            execution_cmd += f" {param}"
        
        # Run command
        os.system(execution_cmd)
    else:
        # Windows execution path
        interpreter = "python"
        script_path = "D:/博一/科研/神经算子/实验/test/wpinns-main/wPINNS.py"
        working_dir = "D:/博一/科研/神经算子/实验/test/wpinns-main"
        
        process = subprocess.Popen([interpreter, script_path] + command_args, cwd=working_dir)
        process.wait()
