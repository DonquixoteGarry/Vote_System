import matplotlib.pyplot as plt

def myplot(example_set,col,row,alert_string,title):
    img_iter = 0
    len_example = len(example_set)
    step = len_example // (col * row)
    fig=plt.figure(figsize=(col, row))
    fig.canvas.set_window_title(title)
    for order in range(0, col * row):
        if step == 0:
            step = 1
        if order >= len_example:
            break
        plt.subplot(col, row, order + 1)
        plt.xticks([], [])
        plt.yticks([], [])
        ex, ori, fault, = example_set[img_iter]
        img_iter += step
        plt.title("{} -> {}".format(ori, fault))
        plt.imshow(ex, cmap="gray")
    plt.suptitle(alert_string+' ( full size = {} )'.format(len_example))

    plt.tight_layout()
    plt.show()