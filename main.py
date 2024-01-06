import subprocess

processes = [
    ['python3', './src/clean_scale.py'],
    ['python3', './src/feature_select_reduce.py'],
    ['python3', './src/model.py'],
]

if __name__ == '__main__':
    input = input('Only run clean_and_scale.py? (Y/n)')
    try:
        if input == 'Y' or input == 'y' or input == '':
            subprocess.run(processes[0])
        elif input == 'n' or input == 'N':
            input = input('Are you sure? It will take some time... (Y/n)')
            if input == 'Y' or input == 'y' or input == '':
                for process in processes:
                    subprocess.run(process)
            elif input == 'n' or input == 'N':
                subprocess.run(processes[0])
            else:
                print('Operation failed. Please input only y or n')
        else:
            print('Operation failed. Please input only y or n')
    except Exception as e:
        print(e)