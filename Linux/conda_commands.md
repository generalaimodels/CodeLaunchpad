| **Command**               | **Generalized Format**                    | **Real-time Example**                                |
|---------------------------|-------------------------------------------|-----------------------------------------------------|
| **Check Conda version**   | `conda --version`                         | `conda --version`                                  |
| **Update Conda**         | `conda update -n base -c defaults conda`  | `conda update -n base -c defaults conda`          |
| **Create a new env**     | `conda create -n <env_name> <packages>`    | `conda create -n myenv python=3.10`               |
| **Activate an env**      | `conda activate <env_name>`                | `conda activate myenv`                            |
| **Deactivate an env**    | `conda deactivate`                         | `conda deactivate`                                |
| **List all envs**        | `conda env list` or `conda info --envs`    | `conda env list`                                  |
| **Remove an env**        | `conda remove --name <env_name> --all`     | `conda remove --name myenv --all`                 |
| **List all installed packages** | `conda list`                         | `conda list`                                      |
| **Install a package**    | `conda install -n <env_name> <package>`    | `conda install -n myenv numpy`                    |
| **Install package in active env** | `conda install <package>`       | `conda install pandas`                            |
| **Install from a specific channel** | `conda install -c <channel> <package>` | `conda install -c conda-forge matplotlib` |
| **Remove a package**     | `conda remove -n <env_name> <package>`     | `conda remove -n myenv scipy`                     |
| **Search for a package** | `conda search <package>`                   | `conda search tensorflow`                         |
| **Update a package**     | `conda update -n <env_name> <package>`     | `conda update -n myenv scikit-learn`              |
| **Update all packages**  | `conda update --all`                       | `conda update --all`                              |
| **Check environment details** | `conda info`                         | `conda info`                                      |
| **Clone an environment** | `conda create --name <new_env> --clone <existing_env>` | `conda create --name myenv_copy --clone myenv` |
| **Export environment**   | `conda env export > <filename>.yml`        | `conda env export > myenv.yml`                    |
| **Create env from YAML** | `conda env create -f <filename>.yml`       | `conda env create -f myenv.yml`                   |
| **Remove unused packages** | `conda clean --all`                     | `conda clean --all`                               |
| **Check dependency tree** | `conda info --dependencies`               | `conda info --dependencies`                       |
| **List available channels** | `conda config --show channels`         | `conda config --show channels`                    |
| **Add a new channel**    | `conda config --add channels <channel>`    | `conda config --add channels conda-forge`         |
| **Remove a channel**     | `conda config --remove channels <channel>` | `conda config --remove channels conda-forge`      |
| **Check current Conda config** | `conda config --show`              | `conda config --show`                             |
