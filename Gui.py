import tkinter as tk
from tkinter import ttk
import matplotlib
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

matplotlib.use("TkAgg")  # Use the Tkinter backend for matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ValuesDict import *
from ParseText import *
from ClusterUsingWord2Vec import cluster_word2vec

welcome_text = "Language of Genesis"
welcome_subtitle = "Work of Sivan Sborovsky"
explanation_text = "Watch as the first book of the Hebrew Bible breaks down to its very atoms"
next_button_text = "I'm Ready"
additional_info_init_text = "Cluster to see more details"
cluster_value_text = "Cluster by Value"
cluster_context_text = "Cluster by Context"
cluster_statistics_text = "Cluster by Statistics"
font = "Rubik"
buttons_size = 20
# font = "Rubik light"


file_dict = {"Texts/1-3:Bereshit.txt":"בראשית", "Texts/3-6:GanEdenDownfall.txt":"גירוש גן עדן",
             "Texts/6-11:Noah.txt": "נוח", "Texts/12-13:God_Blesses_Abraham.txt": "ברכת אברהם",
             "Texts/14-17:LechLechaPart2.txt": "אברהם מקיים את ההבטחה" , "Texts/18-22:sdomVeamora.txt":"סדום ועמורה",
             "Texts/23-25:sara.txt":"שרה", "Texts/25-28:Toldot.txt":"עשו ויעקב", "Texts/28-32:yaakov.txt":"יעקב",
             "Texts/32-36:vayshlach.txt":"מפגש עם עשו", "Texts/37-40:vayeshev.txt": "יוסף",
             "Texts/41-44:ParoosDreams.txt":"חלומות פרעה","Texts/44-48:Vayegash.txt": "איחוד יוסף ויעקב",
             "Texts/48-50:vayehi.txt":"סיום"
             }


def perform_value_cluster():

    data_folder = "Texts"
    file_paths = []
    for filename in os.listdir(data_folder):
        file_paths.append("Texts/" + filename)

    # List of dictionaries
    dictionaries = get_dicts()
    dict_names = get_dicts_names()
    # Initialize the vector array
    num_files = len(file_paths)
    num_dicts = len(dictionaries)
    dataset = np.zeros((num_files, num_dicts))

    # Iterate over the files
    for file_index, file_path in enumerate(file_paths):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            words = text.split()
            words_count = len(words)

            # Iterate over the words
            for word in words:
                # Check if the word belongs to any of the dictionaries
                for dict_index, dictionary in enumerate(dictionaries):
                    word = word.strip(".,;:\r\n")
                    if word in dictionary:
                        # Increment the corresponding element in the vector array
                        # value = 0.0
                        # value = float(1/words_count)
                        dataset[file_index, dict_index] += 1


    # Perform clustering using K-means
    num_clusters = 5  # Set the number of clusters
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(dataset)

    # Store file paths and vectors for each cluster
    clusters = [[] for _ in range(num_clusters)]
    cluster_vectors = [[] for _ in range(num_clusters)]
    for file_index, cluster_label in enumerate(cluster_labels):
        clusters[cluster_label].append(file_paths[file_index])
        cluster_vectors[cluster_label].append(dataset[file_index])

    # Update additional_info_text with cluster information
    def perform_value_cluster():
        # Rest of your code...
        # Update additional_info_text with cluster information

        # Create a table view for the additional information
        additional_info_table = ttk.Treeview(root, columns=("Cluster", "Files", "Averages"), show="headings")
        additional_info_table.heading("Cluster", text="Cluster")
        additional_info_table.heading("Files", text="Files")
        additional_info_table.heading("Averages", text="Averages")

        # Insert cluster information into the table
        for cluster_index, (cluster_files, cluster_vector_list) in enumerate(zip(clusters, cluster_vectors)):
            cluster_files_str = "\n".join([file_dict[str(file_path)] for file_path in cluster_files])
            cluster_vectors_avg = np.mean(cluster_vector_list, axis=0)
            cluster_averages_str = "\n".join([f"{dict_names[dict_index]} = {avg_value:.3f}" for dict_index, avg_value in
                                              enumerate(cluster_vectors_avg)])

            additional_info_table.insert("", "end", values=(cluster_index, cluster_files_str, cluster_averages_str))

        # Clear the existing table
        additional_info_table_frame = ttk.Frame(root)
        additional_info_table_frame.pack(padx=20)
        for child in additional_info_table_frame.winfo_children():
            child.destroy()

        # Display the table
        additional_info_table.pack(in_=additional_info_table_frame, side="left", padx=20, pady=20)

        # Configure scrollbar for the table
        table_scrollbar = ttk.Scrollbar(additional_info_table_frame, orient="vertical",
                                        command=additional_info_table.yview)
        additional_info_table.configure(yscroll=table_scrollbar.set)
        table_scrollbar.pack(side="right", fill="y")

    additional_info_text.delete("1.0", tk.END)  # Clear the text widget


    # Print the file paths and average values for each cluster
    for cluster_index, (cluster_files, cluster_vector_list) in enumerate(zip(clusters, cluster_vectors)):
        additional_info_text.insert(tk.END, f"Cluster {cluster_index}\n")
        additional_info_text.insert(tk.END, "Files in Cluster:\n")
        for file_path in cluster_files:
            story_title = file_dict[str(file_path)]
            additional_info_text.insert(tk.END, f"{story_title}\n")
        additional_info_text.insert(tk.END, "Averages:\n")

        # Calculate the average vector for the cluster
        cluster_vectors_avg = np.mean(cluster_vector_list, axis=0)
        for dict_index, avg_value in enumerate(cluster_vectors_avg):
            parameter_name = dict_names[dict_index]
            additional_info_text.insert(tk.END, f"{parameter_name} = {avg_value:.3f}\n")
        additional_info_text.insert(tk.END, "\n")

    cluster_plot.clear()
    # Plot the clusters for value-based clustering
    cluster_plot.scatter(dataset[:, 0], dataset[:, 1], c=cluster_labels, cmap='viridis')
    cluster_plot.set_title("Value Clustering")
    cluster_plot.set_xlabel("Value 1")
    cluster_plot.set_ylabel("Value 2")
    cluster_canvas.draw()


def perform_context_cluster():
    cluster_plot.clear()
    cluster_word2vec()
    num_points = 100
    np.random.seed(0)
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    labels = np.random.randint(0, 3, num_points)

    cluster_plot.clear()
    # Plot the clusters for context-based clustering (using random data)
    cluster_plot.scatter(x, y, c=labels, cmap='viridis')
    cluster_plot.set_title("Context Clustering")
    cluster_plot.set_xlabel("X-axis Label")  # Customize your X-axis label
    cluster_plot.set_ylabel("Y-axis Label")  # Customize your Y-axis label
    cluster_canvas.draw()


def perform_statistics_cluster():
    data_folder = "Texts"
    filenames = os.listdir(data_folder)
    documents = []
    for filename in os.listdir(data_folder):
        with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as file:
            # document = file.read()
            documents.append(parse_file(filename))

    # Join the parsed content of each file into a single string
    documents = [' '.join(doc) for doc in documents]

    # Step 2: Word Embedding using Word2Vec
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)


    # Option 1: TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)


    # Step 3: Document Representation
    # Option 1: TF-IDF
    document_vectors = X_tfidf.toarray()

    data_folder = "Texts"
    file_paths = []
    for filename in os.listdir(data_folder):
        file_paths.append("Texts/" + filename)

    # List of dictionaries
    dictionaries = get_dicts()
    dict_names = get_dicts_names()
    # Initialize the vector array
    num_files = len(file_paths)
    num_dicts = len(dictionaries)
    dataset = np.zeros((num_files, num_dicts))

    # Iterate over the files
    for file_index, file_path in enumerate(file_paths):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            words = text.split()
            words_count = len(words)

            # Iterate over the words
            for word in words:
                # Check if the word belongs to any of the dictionaries
                for dict_index, dictionary in enumerate(dictionaries):
                    word = word.strip(".,;:\r\n")
                    if word in dictionary:
                        # Increment the corresponding element in the vector array
                        # value = 0.0
                        # value = float(1/words_count)
                        dataset[file_index, dict_index] += 1

    # Perform clustering using K-means
    num_clusters = 5  # Set the number of clusters
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(document_vectors)

    # Step 5: Evaluation and Interpretation
    # Visualize Clusters
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, perplexity=10)
    transformed_vectors = pca.fit_transform(document_vectors)  # Use this one
    # Plot the clusters
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], c=cluster_labels, cmap='viridis')
    plt.title("Cluster by Statistics")
    plt.xlabel("")
    plt.ylabel("")
    # plt.clabel("Dimension 3")

    plt.show()

    for cluster in range(num_clusters):
        print(f"Cluster {cluster + 1} documents:")
        cluster_indices = np.where(cluster_labels == cluster)[0]
        for index in cluster_indices:
            print(file_dict[filenames[index]])
            print("--------------------")
        print("\n")
    # Store file paths and vectors for each cluster
    clusters = [[] for _ in range(num_clusters)]
    cluster_vectors = [[] for _ in range(num_clusters)]
    for file_index, cluster_label in enumerate(cluster_labels):
        clusters[cluster_label].append(file_paths[file_index])
        cluster_vectors[cluster_label].append(dataset[file_index])

    # Update additional_info_text with cluster information

    additional_info_text.delete("1.0", tk.END)  # Clear the text widget

    # Print the file paths and average values for each cluster
    for cluster_index, (cluster_files, cluster_vector_list) in enumerate(zip(clusters, cluster_vectors)):
        additional_info_text.insert(tk.END, f"Cluster {cluster_index}\n")
        additional_info_text.insert(tk.END, "Files in Cluster:\n")
        for file_path in cluster_files:
            story_title = file_dict[str(file_path)]
            additional_info_text.insert(tk.END, f"{story_title}\n")
        additional_info_text.insert(tk.END, "Averages:\n")

        # Calculate the average vector for the cluster
        cluster_vectors_avg = np.mean(cluster_vector_list, axis=0)
        for dict_index, avg_value in enumerate(cluster_vectors_avg):
            parameter_name = dict_names[dict_index]
            additional_info_text.insert(tk.END, f"{parameter_name} = {avg_value:.3f}\n")
        additional_info_text.insert(tk.END, "\n")

    cluster_plot.clear()
    # Plot the clusters for value-based clustering
    cluster_plot.scatter(dataset[:, 0], dataset[:, 1], c=cluster_labels, cmap='viridis')
    cluster_plot.set_title("Statistics Clustering")
    cluster_plot.set_xlabel("Value 1")
    cluster_plot.set_ylabel("Value 2")
    cluster_canvas.draw()


    # # Clear existing plots
    # cluster_plot_statistics.clear()
    #
    # # Step 1: Data Preprocessing
    # data_folder = "Texts"
    # filenames = os.listdir(data_folder)
    # documents = []
    # for filename in os.listdir(data_folder):
    #     with open(os.path.join(data_folder, filename), "r", encoding="utf-8") as file:
    #         documents.append(parse_file(filename))
    #
    # # Join the parsed content of each file into a single string
    # documents = [' '.join(doc) for doc in documents]
    #
    # # Step 2: Text Vectorization
    # vectorizer = CountVectorizer()
    # X = vectorizer.fit_transform(documents)
    #
    # # Step 3: TF-IDF Transformation
    # tfidf_transformer = TfidfTransformer()
    # X_tfidf = tfidf_transformer.fit_transform(X)
    #
    # # Step 4: Document Representation
    # document_vectors = X_tfidf.toarray()
    #
    # # Step 5: Clustering Algorithm (K-means)
    # num_clusters = 5  # Set the number of clusters
    # kmeans = KMeans(n_clusters=num_clusters)
    # cluster_labels = kmeans.fit_predict(document_vectors)
    #
    # # Step 6: Visualization
    # pca = PCA(n_components=2)
    # transformed_vectors = pca.fit_transform(document_vectors)
    #
    # # Plot the clusters
    # cluster_plot_statistics.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1],
    #                                 c=cluster_labels, cmap='viridis')
    # cluster_plot_statistics.set_title("Statistics Clustering")
    # cluster_plot_statistics.set_xlabel("Dimension 1")
    # cluster_plot_statistics.set_ylabel("Dimension 2")
    #
    # # Display the plot
    # cluster_canvas_statistics.draw()
    #
    # # Print cluster information
    # for cluster in range(num_clusters):
    #     print(f"Cluster {cluster + 1} documents:")
    #     cluster_indices = np.where(cluster_labels == cluster)[0]
    #     for index in cluster_indices:
    #         print(filenames[index])
    #         print("--------------------")
    #     print("\n")
    #
    # plt.show()
    #
    # cluster_plot.clear()
    # cluster_plot.scatter(dataset[:, 0], dataset[:, 1], c=cluster_labels, cmap='viridis')
    # cluster_plot.set_title("Value Clustering")
    # cluster_plot.set_xlabel("Value 1")
    # cluster_plot.set_ylabel("Value 2")
    # cluster_canvas.draw()



def open_screen():
    welcome_label = tk.Label(root, text=welcome_text, font=(font, 120))
    welcome_label.pack(pady=50)
    welcome_sublabel = tk.Label(root, text=welcome_subtitle, font=(font, 50 ))
    welcome_sublabel.pack(pady=20)

    explanation_label = tk.Label(root,
                                 text= explanation_text,
                                 font=(font, 40))
    explanation_label.pack(pady=20)

    next_button = tk.Button(root, text=next_button_text, font=(font, 40), command=main_menu)
    next_button.pack(pady=20)


def main_menu():
    # Clear the open screen
    for widget in root.winfo_children():
        widget.destroy()

    cluster_value_button = tk.Button(root, text=cluster_value_text, font=(font, buttons_size), command=perform_value_cluster)
    cluster_value_button.pack(side="top", padx=20)

    cluster_context_button = tk.Button(root, text=cluster_context_text, font=(font, buttons_size), command=perform_context_cluster)
    cluster_context_button.pack(side="top", padx=20)

    cluster_statistics_button = tk.Button(root, text=cluster_statistics_text, font=(font, buttons_size),
                                       command=perform_statistics_cluster)
    cluster_statistics_button.pack(side="top", padx=20)

    additional_info_button = tk.Button(root, text="Additional Info", font=(font, buttons_size), command=toggle_additional_info)
    additional_info_button.pack(pady=20)

    global additional_info_text  # Declare as a global variable
    additional_info_text = tk.Text(root, font=(font, 25), wrap="word", height=20, width=60, bg = "light blue")
    additional_info_text.insert("1.0", additional_info_init_text)
    additional_info_text.pack(side="left", padx=33)

    # Create a figure for the plot
    global cluster_plot
    fig = Figure(figsize=(8, 6), dpi=100)
    cluster_plot = fig.add_subplot(111)

    # Create a canvas for the plot
    global cluster_canvas
    cluster_canvas = FigureCanvasTkAgg(fig, master=root)
    cluster_canvas.get_tk_widget().pack(side="right", padx=20)

    # Create a figure for the context-based clustering plot
    fig_context = Figure(figsize=(8, 6), dpi=100)
    global cluster_plot_context
    cluster_plot_context = fig_context.add_subplot(111)

    # Create a canvas for the context-based clustering plot
    global cluster_canvas_context
    cluster_canvas_context = FigureCanvasTkAgg(fig_context, master=root)
    cluster_canvas_context.get_tk_widget().pack(side="right", padx=20)

    # Create a figure for the statistics-based clustering plot
    fig_statistics = Figure(figsize=(8, 6), dpi=100)
    global cluster_plot_statistics
    cluster_plot_statistics = fig_statistics.add_subplot(111)

    # Create a canvas for the statistics-based clustering plot
    global cluster_canvas_statistics
    cluster_canvas_statistics = FigureCanvasTkAgg(fig_statistics, master=root)
    cluster_canvas_statistics.get_tk_widget().pack(side="right", padx=20)


def toggle_additional_info():
    if additional_info_text.winfo_ismapped():
        additional_info_text.pack_forget()
    else:
        additional_info_text.pack(side="right", padx=20)


# Create the root window
root = tk.Tk()
root.geometry("1920x1080")

# Start with the open screen
open_screen()

# Run the GUI event loop
root.mainloop()