'''Some brief instructions to reinstall my own packagesI have renamed the Functions folder ocean_functions where the o_functions module contains what I need to complete my studies. to redo over the function which currently installed in the environment: geo_env_test'''conda activate geo_env_test- navigate to the location of the setup.py filecd /Volumes/PD/GitHub/python-oceanography/Delft 3D FM Suite 2019/ocean_functions(geo_env_test) af@unix dist % python setup.py bdist_wheel                      - then uninstall the package/install the package via the file located in the dist folder in ocean_functionspip uninstall o_functions-0.1.0-py3-none-any.whl     pip install o_functions-0.1.0-py3-none-any.whl     created alias so now on mac you only need to run $ func_owhich reinstalls the package for you for initial testing#############################################################data_prepkit: tools to cut and process the data into a usable format for a wide range of statistical and comparative analysisdata_metrics: tools to provide statistical analsis to produce viable results across all code.#############################################################