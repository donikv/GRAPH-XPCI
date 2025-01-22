import yaml
import subprocess
import shlex
import sys 
import argparse

def parse_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def format_args(args):
    formatted_args = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                formatted_args.append(f'--{key}')
        elif isinstance(value, list):
            formatted_args.append(f'--{key} {" ".join(value)}')
        else:
            formatted_args.append(f'--{key}={shlex.quote(str(value))}')
    return formatted_args

def call_python_file(file_path, args):
    command = f"python {file_path} {' '.join(args)}"
    subprocess.run(command, shell=True)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_path', help='Path to the YAML file')
    parser.add_argument('--args', nargs='*', help='Additional arguments')
    args = parser.parse_args()

    # Parse the YAML file
    data = parse_yaml(args.yaml_path)

    # Update the arguments from the YAML file with the additional arguments
    if args.args:
        for arg in args.args:
            key, value = arg.split('=')
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            data['args'][key] = value

    # Format the arguments
    formatted_args = format_args(data['args'])

    # Call the Python file with the updated arguments
    call_python_file(data['python_file'], formatted_args)