import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def violin_plot(scores,save_name,is_save = True):
    '''
    Scores should be like this:
    {'Bleu1':[] 'Bleu2':[],}
    '''
    df = pd.DataFrame(scores)

    sns.set(style="whitegrid")  # Optional: Set the plot style

    plt.figure(figsize=(10, 6))  # Optional: Adjust the figure size

    sns.violinplot(data=df, inner="points")

    # Optional: Add labels and a title
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title(save_name)
    save_name = save_name+".png"
    if is_save:
        plt.savefig(save_name)


if  __name__ == "__main__":
    data_dict = {
        'Column1': [1, 2, 3, 4, 5],
        'Column2': [6, 7, 8, 9, 10],
        'Column3': [11, 12, 13, 14, 15],
        'Column4': [10, 7, 8, 39, 10],
        'Column5': [11, 1, 13, 24, 15],
        'Column6': [6, 7, 8, 19, 10],
        'Column7': [11, 12, 3, 14, 15]
    }

    violin_plot(data_dict,"abc")