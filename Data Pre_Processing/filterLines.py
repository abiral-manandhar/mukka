def remove_unwanted_lines(input_file_path, output_file_path):
    # Open the input file in read mode and the output file in write mode
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        # Iterate through each line in the input file
        for line in infile:
            # Write lines that do not contain the unwanted text to the output file
            if "Received data does not contain all required lines." not in line:
                outfile.write(line)


if __name__ == "__main__":
    # Define the input and output file paths
    input_file = 'output_log.txt'
    output_file = '../Data/Stationary/cleaned_output_log.txt'

    # Call the function to remove unwanted lines
    remove_unwanted_lines(input_file, output_file)
