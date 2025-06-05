import matplotlib.pyplot as plt
def class_distribution(ds, color_palette, title='Test class distribution', column='labels'):
    s = ds['labels'].value_counts()
    labels = ds[column].unique()
    plt.pie(
        s,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.3, edgecolor='black', linewidth=1),
        labels=None,
        colors=color_palette[0:2],
        pctdistance=1.12,)
    plt.title(title)
    plt.axis('equal')
    plt.legend(labels=labels, loc='upper left')
    plt.show()