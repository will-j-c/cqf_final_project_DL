time_taken = 100
run_name = 'string'

lines = [f'Time taken: {time_taken}', f'Run name: {run_name}']

with open('log.txt', 'w') as f:
    f.writelines('\n'.join(lines))