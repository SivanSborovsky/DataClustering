import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from ParseText import *

file_dict = {"1-3:Bereshit.txt":"בראשית", "3-6:GanEdenDownfall.txt":"גירוש גן עדן",
             "6-11:Noah.txt": "נוח", "12-13:God_Blesses_Abraham.txt": "ברכת אברהם",
             "14-17:LechLechaPart2.txt": "אברהם מקיים את ההבטחה" , "18-22:sdomVeamora.txt":"סדום ועמורה",
             "23-25:sara.txt":"שרה", "25-28:Toldot.txt":"עשו ויעקב", "28-32:yaakov.txt":"יעקב",
             "32-36:vayshlach.txt":"מפגש עם עשו", "37-40:vayeshev.txt": "יוסף",
             "41-44:ParoosDreams.txt":"חלומות פרעה","44-48:Vayegash.txt": "איחוד יוסף ויעקב",
             "48-50:vayehi.txt":"סיום"
             }

# Step 1: Data Preprocessing
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

# Rest of your code...

# Option 1: TF-IDF
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X)

# Option 2: Word2Vec
# tokenized_documents = [document.split() for document in documents]
# model = Word2Vec(tokenized_documents, min_count=1)

# Step 3: Document Representation
# Option 1: TF-IDF
document_vectors = X_tfidf.toarray()

# Option 2: Word2Vec
# document_vectors = np.array([np.mean([model.wv[word] for word in document.split()], axis=0) for document in documents])

# Rest of your code...

# Step 4: Clustering Algorithm (K-means)
num_clusters = 5  # Set the number of clusters
kmeans = KMeans(n_clusters=num_clusters)
cluster_labels = kmeans.fit_predict(document_vectors)

# Step 5: Evaluation and Interpretation
# Visualize Clusters
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, perplexity=10)

# Uncomment either PCA or t-SNE for visualization
transformed_vectors = pca.fit_transform(document_vectors)  # Use this one

# transformed_vectors = tsne.fit_transform(document_vectors)

# Plot the clusters
plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], c=cluster_labels, cmap='viridis')
plt.title("Document Clustering")
plt.xlabel("")
plt.ylabel("")
# plt.clabel("Dimension 3")

plt.show()

for cluster in range(num_clusters):
    print(f"Cluster {cluster+1} documents:")
    cluster_indices = np.where(cluster_labels == cluster)[0]
    for index in cluster_indices:
        print(file_dict[filenames[index]])
        print("--------------------")
    print("\n")