import subprocess

# Define the ranges or lists for N, br, and bc

# Iterate over the values and run the command
with open(f"output_test.txt", "w") as file:
    for N in [1, 3, 101, 500, 1024, 2047, 3111] :
        # Construct the command
        command = f"python3 gpt149.py -N {N} part1"

        # Run the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Save the output to a file
        file.write("-------------Part1---------------")
        file.write(f"N = {N} \n")
        file.write(stdout.decode())
                
        if stderr:
            file.write("\nErrors:\n")
            file.write(stderr.decode())

        # Construct the command
        command = f"python3 gpt149.py -N {N} part2"

        # Run the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Save the output to a file
        file.write("-------------Part2---------------")
        file.write(f"N = {N} \n")
        file.write(stdout.decode())
                
        if stderr:
            file.write("\nErrors:\n")
            file.write(stderr.decode())


        # Construct the command
        command = f"python3 gpt149.py -N {N} part3"

        # Run the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Save the output to a file
        file.write("-------------Part3---------------")
        file.write(f"N = {N} \n")
        file.write(stdout.decode())
                
        if stderr:
            file.write("\nErrors:\n")
            file.write(stderr.decode())
print("Experiments completed and outputs stored.")
