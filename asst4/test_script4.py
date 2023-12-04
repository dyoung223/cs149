import subprocess

# Define the ranges or lists for N, br, and bc

# Iterate over the values and run the command
with open(f"output_test.txt", "w") as file:
    for case in [(10, 2, 2), (1, 1, 1), (20, 25, 26), (10, 3, 3), (4, 2, 8), (1000, 101, 121), (1000, 500, 500), (1024, 512, 513), (2000, 100, 1000), (10, 1, 1)] :
        N = case[0]
        br = case[1]
        bc = case[2]
        # Construct the command
        command = f"python3 gpt149.py -N {N} -br {br} -bc {bc} part4"

        # Run the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Save the output to a file
        file.write(f"N = {N}, br = {br}, bc = {bc} \n")
        file.write(stdout.decode())
                
        if stderr:
            file.write("\nErrors:\n")
            file.write(stderr.decode())

print("Experiments completed and outputs stored.")
