import subprocess
import argparse

def compile(args):
    command = "{} {}".format("srun -p mpi-homo-short -A nprg042s", args.compile)
    parts = command.split(" ")
    print(command)
    subprocess.run(parts)

def benchmark(args):
    if args.test:
        mapper = {
            "01" : ["01-32k.A", "01-32k.B"], "02" : ["02-64k.A", "02-64k.B"], 
            "03" : ["03-128k.A", "03-128k.B"], "04" : ["04-64k.A", "04-128k.B"],
            "05" :  ["05-128k.A", "05-64k.B"], "10" : ["10-1M.A", "10-1M.B"]
        }
        args.file1 =  mapper[args.test][0]
        args.file2 = mapper[args.test][1]
    #command = "srun -p mpi-homo-short -A nprg042s ./levenshtein_serial ../data/{} ../data/{}".format(args.file1, args.file2)
    if args.serial:
        command = "srun -p mpi-homo-short -A nprg042s ./levenshtein_serial ../data/{} ../data/{}".format(args.file1, args.file2)
        serial_parts = command.split(" ")
        print("Serial")
        print(command)
        subprocess.run(serial_parts)

    print("Parallel")
    if args.cpu: # args.k
        command = "srun -p mpi-homo-short -A nprg042s -c {} ./levenshtein ../data/{} ../data/{}".format(
            args.cpu, args.file1, args.file2
        )
        print(command)
        parts = command.split(" ")
        subprocess.run(parts)
    else:
        """
        cpus = [32, 64]
        for cpu in cpus:
            command = "srun -p mpi-homo-short -A nprg042s -c {} ./levenshtein ../data/{} ../data/{}".format(cpu, args.file1, args.file2)
            print(command)
            parts = command.split(" ")
            subprocess.run(parts)
        """
        command = "./levenshtein ../data/{} ../data/{}".format(args.file1, args.file2)
        print(command)
        parts = command.split(" ")
        subprocess.run(parts)

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_options()
    
    def set_options(self):
        self.parser.add_argument("--test", default="01", type=str)
        self.parser.add_argument("--file1", default=None, type=str)
        self.parser.add_argument("--file2", default=None, type=str)
        self.parser.add_argument('--benchmark', default=False, action='store_true')
        self.parser.add_argument('--serial', default=False, action='store_true')
        self.parser.add_argument("--compile", default=None, type=str)
        self.parser.add_argument("--cpu", default=None, type=int)
    
    def get_arguments(self):
        return self.parser.parse_args()

if __name__ == "__main__":
    parser = Parser()
    args = parser.get_arguments()

    if args.compile:
        compile(args)
    elif args.benchmark:
        benchmark(args)