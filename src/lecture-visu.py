import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

data_df = pd.read_csv("data\Country-data.csv")
print(data_df)
print(data_df.mean()) # Moyenne
print(data_df.std()) # Variance
print(data_df.corr()) # Matrice de corrélation

# Paramètres
step = 10  # Nombre de pays affichés par page
total_countries = len(data_df)

for feature in data_df.columns[1:]:
    sorted_df = data_df.sort_values(by=feature, ascending=False)
    #sorted_df = pd.concat([sorted_df.head(3), sorted_df.tail(3)])

    """   plt.figure(figsize=(20, 6))  # Plus large
    sns.barplot(data=sorted_df, x="country", y=feature)
    
    plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotation et réduction de taille
    plt.title(f"Barplot of {feature} by Country")

    plt.tight_layout()  # Ajuste les marges
    plt.show()"""


    # Initialiser la fenêtre Tkinter
    root = tk.Tk()
    root.title("Histogramme interactif avec Slider")

    # Créer une figure Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))

    def update_plot(start_index):
        """ Met à jour le graphique en fonction du slider """
        ax.clear()
        subset = sorted_df.iloc[start_index:start_index + step]
        sns.barplot(data=subset, x="country", y=feature, ax=ax)
        ax.set_xticklabels(subset["country"], rotation=45)
        ax.set_title(f"{feature} Rate ({start_index + 1} - {start_index + step})")
        canvas.draw()

    # Canvas Matplotlib dans Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    # Créer un slider Tkinter
    slider = tk.Scale(root, from_=0, to=total_countries - step, orient="horizontal",
                    length=600, resolution=step, label="Début:", command=lambda val: update_plot(int(val)))
    slider.pack()

    # Afficher le premier graphique
    update_plot(0)

    # Lancer l'interface Tkinter
    root.mainloop()
