import matplotlib.pyplot as plt
from regex_for_log import get_both_loss
# Define the x and y values
adam, sophia = get_both_loss()

def draw_line(plt, loss_data, line_name):
    x = [int(i[0]) * 0.1 for i in loss_data]
    y = [float(i[1]) for i in loss_data]

    # Plot the curve
    plt.plot(x, y, label=line_name)

draw_line(plt, adam, "Adam")
draw_line(plt, sophia, "Sophia")

# Add labels and title
plt.xlabel('step')
plt.ylabel('loss')
plt.title('Training Loss')

plt.legend()
# Show the plot
plt.show()