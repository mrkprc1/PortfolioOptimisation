import matplotlib.pyplot as plt 

def plot_mat(matrix, values=True):
    n = matrix.shape[0]
    fig, ax = plt.subplots()
    

    if values:
        ax.matshow(matrix[tuple([n-i for i in range(1, n+1)]),:], cmap=plt.cm.Blues)
        min_val, max_val = 0, n
        for i in range(n):
            for j in range(n):
                ax.text(i, j, str(matrix[n-j-1,i]), va='center', ha='center')
        ax.set_xlim(-0.5, n-0.5)
        ax.set_ylim(-0.5, n-0.5)
        ax.set_xticks(np.arange(max_val))
        ax.set_yticks(np.arange(max_val))
    else:
        ax.matshow(matrix, cmap=plt.cm.Blues)

        plt.show()