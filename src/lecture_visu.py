import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

df = pd.read_csv("data\Country-data.csv")
print(df)
print(df.mean(numeric_only=True))  # Moyenne
print(df.std(numeric_only=True))  # Variance
print(df.corr(numeric_only=True))  # Matrice de corrélation

###################################
#     Visualisation Globale       #
###################################

def visu_total(df):
    # Paramètres
    step = 10  # Nombre de pays affichés par page
    total_countries = len(df)

    for feature in df.columns[1:]:
        sorted_df = df.sort_values(by=feature, ascending=False)
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

#visu_total(df)

###################################
#        Visualisation top        #
###################################


# Initialisation de l'application Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Visualisation des Indicateurs Économiques"),
    
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns if col != 'country'],
        value='gdpp',
        clearable=False
    ),
    
    html.Div([
        html.Div([dcc.Graph(id='top-5-graph')], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(id='bottom-5-graph')], style={'width': '50%', 'display': 'inline-block'})
    ])
])

@app.callback(
    [Output('top-5-graph', 'figure'), Output('bottom-5-graph', 'figure')],
    [Input('feature-dropdown', 'value')]
)
def update_graphs(selected_feature):
    top_5 = df.nlargest(5, selected_feature)
    bottom_5 = df.nsmallest(5, selected_feature).sort_values(by=selected_feature, ascending=False)
    
    top_fig = px.bar(top_5, x='country', y=selected_feature, title=f'Top 5 - {selected_feature}')
    bottom_fig = px.bar(bottom_5, x='country', y=selected_feature, title=f'Bottom 5 - {selected_feature}')
    
    return top_fig, bottom_fig

if __name__ == '__main__':
    app.run_server(debug=True)