import numpy as np
import cv2
import glob
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def ex3_dbscan_supervised():
    """
    DBSCAN pentru clasificare supervizată - versiunea corectă
    Simulează procesul K-means dar cu DBSCAN
    """
    # Încărcarea imaginilor (codul tău original)
    trainCleanImages = [cv2.imread(file) for file in glob.glob("./train/clean/*.png")]
    trainMessyImages = [cv2.imread(file) for file in glob.glob("./train/messy/*.png")]
    testImages = [cv2.imread(file) for file in glob.glob("./test/*.png")]
    trainCleanImages = [cv2.cvtColor(file, cv2.COLOR_BGR2GRAY) for file in trainCleanImages]
    trainMessyImages = [cv2.cvtColor(file, cv2.COLOR_BGR2GRAY) for file in trainMessyImages]
    testImages = [cv2.cvtColor(file, cv2.COLOR_BGR2GRAY) for file in testImages]

    # Extragerea histogramelor (codul tău original)
    numTrain = len(trainCleanImages + trainMessyImages)
    numTest = len(testImages)
    
    cleanTrainData = np.zeros((len(trainCleanImages), 256)).astype(int)
    for idx in range(len(trainCleanImages)):
        hist, bins = np.histogram(trainCleanImages[idx], 256, [0, 256])
        cleanTrainData[idx] = hist
        
    messyTrainData = np.zeros((len(trainMessyImages), 256)).astype(int)
    for idx in range(len(trainMessyImages)):
        hist, bins = np.histogram(trainMessyImages[idx], 256, [0, 256])
        messyTrainData[idx] = hist
        
    X_train = np.concatenate((cleanTrainData, messyTrainData), axis=0)
    
    testData = np.zeros((len(testImages), 256)).astype(int)
    for idx in range(len(testImages)):
        hist, bins = np.histogram(testImages[idx], 256, [0, 256])
        testData[idx] = hist
    X_test = testData
    
    # Etichetele adevărate pentru train
    y_train = np.zeros(len(trainCleanImages) + len(trainMessyImages)).astype(int)
    y_train[len(trainCleanImages):] = 1
    
    # Etichetele adevărate pentru test
    y_test = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0])
    
    print("=== DBSCAN pentru Clasificare Supervizată ===")
    print(f"Date train: {len(X_train)} imagini ({len(trainCleanImages)} clean, {len(trainMessyImages)} messy)")
    print(f"Date test: {len(X_test)} imagini")
    
    # Standardizarea datelor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # IMPORTANT: transform, nu fit_transform!
    
    # ====================================================================
    # PASUL 1: DBSCAN pe datele de antrenare pentru a găsi clustere
    # ====================================================================
    
    def find_best_dbscan_params(X_train_scaled, y_train):
        """Găsește cei mai buni parametri DBSCAN"""
        best_accuracy = 0
        best_params = None
        best_cluster_to_class = None
        best_clusters = None
        
        # Testează diferite combinații de parametri
        eps_values = np.arange(10, 50, 5)
        min_samples_values = [2, 3, 4, 5]
        
        print("\\nCăutarea parametrilor optimi...")
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                train_clusters = dbscan.fit_predict(X_train_scaled)
                
                # Evaluează calitatea clusterelor
                unique_clusters = np.unique(train_clusters)
                noise_points = np.sum(train_clusters == -1)
                valid_clusters = unique_clusters[unique_clusters != -1]
                
                if len(valid_clusters) == 0:
                    continue  # Toate punctele sunt zgomot
                
                # ====================================================================
                # PASUL 2: Creează matricea de frecvență (ca la K-means)
                # ====================================================================
                n_clusters = len(valid_clusters)
                n_classes = 2  # clean (0) și messy (1)
                
                freq_matrix = np.zeros((n_clusters, n_classes), dtype=int)
                
                # Mapare cluster_id la index pentru matricea de frecvență
                cluster_to_index = {cluster_id: idx for idx, cluster_id in enumerate(valid_clusters)}
                
                for i in range(len(X_train_scaled)):
                    if train_clusters[i] != -1:  # Ignoră punctele zgomot
                        cluster_idx = cluster_to_index[train_clusters[i]]
                        true_class = y_train[i]
                        freq_matrix[cluster_idx][true_class] += 1
                
                # ====================================================================
                # PASUL 3: Maparea cluster → clasă (ca la K-means)
                # ====================================================================
                cluster_to_class = {}
                for cluster_id in valid_clusters:
                    cluster_idx = cluster_to_index[cluster_id]
                    # Găsește clasa majoritară pentru acest cluster
                    majority_class = np.argmax(freq_matrix[cluster_idx])
                    cluster_to_class[cluster_id] = majority_class
                
                # ====================================================================
                # PASUL 4: Evaluarea pe datele de antrenare
                # ====================================================================
                train_predictions = np.zeros(len(y_train))
                for i in range(len(y_train)):
                    if train_clusters[i] == -1:
                        # Pentru punctele zgomot, atribuie clasa majoritară
                        train_predictions[i] = 0  # sau poți folosi o strategie mai sofisticată
                    else:
                        train_predictions[i] = cluster_to_class[train_clusters[i]]
                
                accuracy = accuracy_score(y_train, train_predictions)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = (eps, min_samples)
                    best_cluster_to_class = cluster_to_class
                    best_clusters = train_clusters
                    
                    print(f"Nou optim: eps={eps}, min_samples={min_samples}")
                    print(f"  Clustere valide: {len(valid_clusters)}, Zgomot: {noise_points}")
                    print(f"  Acuratețea train: {accuracy:.3f}")
                    print(f"  Matricea frecvență:\\n{freq_matrix}")
                    print(f"  Maparea: {cluster_to_class}")
        
        return best_params, best_cluster_to_class, best_clusters, best_accuracy
    
    # Găsește parametrii optimi
    best_params, cluster_to_class_map, train_clusters, train_accuracy = find_best_dbscan_params(X_train_scaled, y_train)
    
    if best_params is None:
        print("Nu s-au găsit parametri optimi! Încearcă range-uri diferite.")
        return
    
    print(f"\\n=== Rezultate finale cu parametrii optimi ===")
    print(f"Parametri: eps={best_params[0]}, min_samples={best_params[1]}")
    print(f"Acuratețea pe train: {train_accuracy:.3f}")
    
    # ====================================================================
    # PASUL 5: Predicția pe datele de test
    # ====================================================================
    
    # Problema: DBSCAN nu poate prezice direct pe date noi
    # Soluția: Folosim k-NN pentru a găsi cel mai apropiat punct din train
    
    print("\\n=== Predicția pe datele de test ===")
    
    def predict_with_dbscan_knn(X_train_scaled, train_clusters, cluster_to_class_map, X_test_scaled, k=3):
        """
        Prezice clasele pentru datele de test folosind k-NN
        """
        # Folosește k-NN pentru a găsi vecinii cei mai apropiați din train
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X_train_scaled)
        
        # Pentru fiecare punct de test, găsește vecinii din train
        distances, indices = knn.kneighbors(X_test_scaled)
        
        test_predictions = []
        
        for i in range(len(X_test_scaled)):
            # Vecinii cei mai apropiați
            neighbor_indices = indices[i]
            neighbor_clusters = train_clusters[neighbor_indices]
            
            # Votarea majoritară a vecinilor (ignoră zgomotul)
            valid_neighbors = neighbor_clusters[neighbor_clusters != -1]
            
            if len(valid_neighbors) == 0:
                # Toți vecinii sunt zgomot
                prediction = 0  # Clasa implicită
            else:
                # Găsește clusterul cel mai frecvent dintre vecini
                neighbor_classes = [cluster_to_class_map.get(cluster, 0) for cluster in valid_neighbors]
                prediction = max(set(neighbor_classes), key=neighbor_classes.count)
            
            test_predictions.append(prediction)
        
        return np.array(test_predictions)
    
    # Prezice pe test
    test_predictions = predict_with_dbscan_knn(
        X_train_scaled, train_clusters, cluster_to_class_map, X_test_scaled, k=3
    )
    
    # Evaluează rezultatele
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"Predicții test: {test_predictions}")
    print(f"Adevărate test: {y_test}")
    print(f"Acuratețea test: {test_accuracy:.3f} ({100*test_accuracy:.1f}%)")
    
    # Raport detaliat
    print(f"\\n=== Raport de clasificare ===")
    print(classification_report(y_test, test_predictions, target_names=['Clean', 'Messy']))
    
    print(f"\\n=== Matricea de confuzie ===")
    cm = confusion_matrix(y_test, test_predictions)
    print(cm)
    
    # Comparația cu abordarea ta greșită
    print(f"\\n=== Comparație cu abordarea greșită ===")
    print("Abordarea ta originală (greșită):")
    print("- Făcea DBSCAN separat pe train și test")
    print("- Nu crea mapare între clustere și clase")
    print("- Compara direct cluster_id cu class_label")
    print()
    print("Abordarea corectă (ca K-means):")
    print("1. DBSCAN pe train pentru a găsi clustere")
    print("2. Mapare clustere → clase bazată pe frecvență")
    print("3. k-NN pentru predicția pe test (DBSCAN nu poate prezice direct)")
    print("4. Mapare finală cluster → clasă pentru predicții")
    
    return test_predictions, test_accuracy

def demonstrate_difference():
    """
    Demonstrează diferența între abordarea greșită și cea corectă
    """
    print("=== Demonstrație cu date simulate ===")
    
    # Date simulate simple
    np.random.seed(42)
    
    # Train data: 2 clustere naturale pentru fiecare clasă
    clean_cluster1 = np.random.normal([2, 2], 0.5, (50, 2))
    clean_cluster2 = np.random.normal([4, 1], 0.4, (30, 2))
    messy_cluster1 = np.random.normal([7, 6], 0.6, (40, 2))
    messy_cluster2 = np.random.normal([8, 3], 0.5, (35, 2))
    
    X_train_sim = np.vstack([clean_cluster1, clean_cluster2, messy_cluster1, messy_cluster2])
    y_train_sim = np.hstack([np.zeros(80), np.ones(75)])  # 80 clean, 75 messy
    
    # Test data
    X_test_sim = np.random.normal([3, 2], 0.3, (10, 2))  # Ar trebui să fie clean
    y_test_sim = np.zeros(10)
    
    print(f"Date simulate: {len(X_train_sim)} train, {len(X_test_sim)} test")
    
    # DBSCAN pe train
    dbscan = DBSCAN(eps=1.0, min_samples=3)
    train_clusters = dbscan.fit_predict(X_train_sim)
    
    print(f"Clustere găsite: {np.unique(train_clusters)}")
    
    # Matricea de frecvență
    valid_clusters = np.unique(train_clusters[train_clusters != -1])
    if len(valid_clusters) > 0:
        freq_matrix = np.zeros((len(valid_clusters), 2), dtype=int)
        cluster_to_idx = {c: i for i, c in enumerate(valid_clusters)}
        
        for i in range(len(X_train_sim)):
            if train_clusters[i] != -1:
                cluster_idx = cluster_to_idx[train_clusters[i]]
                freq_matrix[cluster_idx][y_train_sim[i]] += 1
        
        print("Matricea de frecvență (clustere x clase):")
        print(freq_matrix)
        
        # Maparea
        cluster_to_class = {}
        for cluster_id in valid_clusters:
            cluster_idx = cluster_to_idx[cluster_id]
            majority_class = np.argmax(freq_matrix[cluster_idx])
            cluster_to_class[cluster_id] = majority_class
            print(f"Cluster {cluster_id} → Clasa {majority_class}")

if __name__ == "__main__":
    # Rulează demonstrația
    demonstrate_difference()
    
    # Rulează clasificarea corectă (decomentează dacă ai datele)
    # ex3_dbscan_supervised()