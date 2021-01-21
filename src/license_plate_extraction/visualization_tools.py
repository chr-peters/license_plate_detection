import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image(img, bounding_box=None):
    fig, ax = plt.subplots()

    ax.imshow(img, cmap="gray")

    if bounding_box is not None:
        x_min, y_min, x_max, y_max = bounding_box

        width = x_max - x_min
        height = y_max - y_min

        rectangle = patches.Rectangle(
            xy=(x_min, y_min),
            width=width,
            height=height,
            edgecolor="red",
            facecolor="none",
            linewidth=1,
        )

        ax.add_patch(rectangle)

    plt.show()