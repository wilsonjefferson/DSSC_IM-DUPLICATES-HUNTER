#!/bin/bash


AML_CloudName="${AML_CloudName:-absent}"
echo "AML_CloudName os env: ${AML_CloudName}"

USERDOMAIN="${USERDOMAIN:-absent}"
echo "USERDOMAIN os env: ${USERDOMAIN}"

project_root=$(pwd)/filtering-of-recognisable-duplicate-tickets/
echo "current directory: ${project_root}"

if [ $AML_CloudName = "AzureCloud" ]; then
    # Set proper HTTP proxies 
    export http_proxy=http://10.141.0.165:3128
    echo "envirnoment variable: "${http_proxy}

    export https_proxy=http://10.141.0.165:3128
    echo "envirnoment variable: "${https_proxy}

    echo "Acquire::http::Proxy \"http://10.141.0.165:3128\";" > /etc/apt/apt.conf.d/proxy.conf
    echo "Acquire::https::Proxy \"http://10.141.0.165:3128\";" >> /etc/apt/apt.conf.d/proxy.conf

    # set nltk folder
    export NLTK_DATA=/home/azureuser/cloudfiles/code/Users/pietro.morichetti/package_data/nltk_data/
    echo "envirnoment variable: "${NLTK_DATA}

    source /anaconda/etc/profile.d/conda.sh  # Load conda into the script's environment

    echo "activate azure-python environment: azureml_py38"
    conda activate azureml_py38
    conda info --envs

    echo "install pip packages..."
    index_url=https://artifactory.sofa.dev/artifactory/api/pypi/pypi-remote/simple
    extra_index_url=https://artifactory.sofa.dev/artifactory/api/pypi/pypi-local/simple

    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} xlrd
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} -U spacy
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} -U pydantic
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} regex
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} huggingface-hub
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} sentence-transformers
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} levenshtein
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} pyspnego
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} ecb-connectors
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} wordcloud
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} pandarallel
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} glob2
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} fasttext
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} bayesian-optimization
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} pybind11
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} matplotlib==3.7.1
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} recordclass
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} cloud-tpu-client
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} mlxtend
    pip install --index-url ${index_url} --extra-index-url ${extra_index_url} scikit-optimize
    echo "pip packages installation completed"
    
    cd ${project_root}/azure_setup
    echo "current directory: $(pwd)"

    sudo ./ml_install.sh
    sudo ./copy_config.sh

    cd $project_root
    echo "current directory: $(pwd)"
fi

echo "extract and move into backup_root folder from project config file..."
config_file=./src/config.yml
backup_root=$(sed -n 's/^\s*backup_root:\s*\(.*\)/\1/p' "$config_file")

cd ${backup_root}
echo "current directory: $(pwd)"

recent_backup=$(ls -dt */ | head -n 1)
echo "recent backup: ${recent_backup}"
echo "model path: ${backup_root}${recent_backup}*/dict_best_model.pickle"
echo "catalog path: ${backup_root}${recent_backup}*/catalog.pickle"

if [ -z "${recent_backup}" ]; then
    echo "WARNING: no backup found"
    echo "run following commands in the terminal:"
    echo "cd ${project_root}"
    echo "python main_train_model_workflow.py"
    echo "python main_build_catalog_workflow.py"
elif [ ! -f ${backup_root}${recent_backup}*/dict_best_model.pickle ]; then 
    echo "WARNING: dict_best_model.pickle file missing"
    echo "run following commands in the terminal:"
    echo "cd ${project_root}"
    echo "python main_train_model_workflow.py"
    echo "python main_build_catalog_workflow.py"
elif [ ! -f ${backup_root}${recent_backup}*/catalog.pickle ]; then 
    echo "WARNING: catalog file missing"
    echo "run following commands in the terminal:"
    echo "cd ${project_root}"
    echo "python main_build_catalog_workflow.py"
else
    cd $project_root
    echo "deploy running..."
    
    python main_deploy_workflow.py
fi

exit 0
