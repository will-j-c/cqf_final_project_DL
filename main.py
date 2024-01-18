import subprocess

processes = [
    ['python3', './src/clean_scale.py'],
    ['python3', './src/feature_select_reduce.py'],
    ['python3', './src/model_selection.py'],
    ['python3', './src/model_tuning.py']
]

if __name__ == '__main__':
    for process in processes:
        subprocess.run(process)