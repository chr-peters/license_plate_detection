import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image(img, bounding_box=None):
    fig, ax = plt.subplots()

    ax.imshow(img, cmap="gray")

    if bounding_box is not None:
        x_min, y_min, width, height = bounding_box

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