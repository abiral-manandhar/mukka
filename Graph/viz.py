import matplotlib.pyplot as plt


# Function to read the data from the text file
def read_data(filename):
    time = []
    accelX_str = []
    accelY_str = []
    accelZ_str = []

    accelX = []
    accelY = []
    accelZ = []

    with open(filename, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            values = line.split(',')
            time.append(float(values[0]))  # Keep timestamp as float
            accelX_str.append(values[1])  # Keep accelX as string
            accelY_str.append(values[2])  # Keep accelY as string
            accelZ_str.append(values[3].strip())  # Keep accelZ as string

            # Convert the string to float for plotting
            accelX.append(float(values[1]))
            accelY.append(float(values[2]))
            accelZ.append(float(values[3].strip()))

    return time, accelX_str, accelY_str, accelZ_str, accelX, accelY, accelZ


# Function to plot the data
def plot_data(time, accelX, accelY, accelZ):
    # Plot accelX vs Time
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, accelX, label='accelX', color='blue')
    plt.xlabel('Time (seconds)')
    plt.ylabel('accelX Reading (float)')
    plt.title('accelX vs Time')
    plt.grid(True)

    # Plot accelY vs Time
    plt.subplot(3, 1, 2)
    plt.plot(time, accelY, label='accelY', color='green')
    plt.xlabel('Time (seconds)')
    plt.ylabel('accelY Reading (float)')
    plt.title('accelY vs Time')
    plt.grid(True)

    # Plot accelZ vs Time
    plt.subplot(3, 1, 3)
    plt.plot(time, accelZ, label='accelZ', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('accelZ Reading (float)')
    plt.title('accelZ vs Time')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Main function
def main():
    filename = '../Data/Stationary/mag_data_active_rot1.csv'  # Replace with your file path
    time, accelX_str, accelY_str, accelZ_str, accelX, accelY, accelZ = read_data(filename)
    plot_data(time, accelX, accelY, accelZ)


if __name__ == "__main__":
    main()
