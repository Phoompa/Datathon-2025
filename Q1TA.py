import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class CoLAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cost of Living Analysis Tool")

        self.filename = None
        
        # Upload CSV Button
        self.upload_button = tk.Button(root, text="Upload CSV", command=self.upload_csv)
        self.upload_button.pack(pady=10)

        # Compute YoY Changes Button
        self.compute_button = tk.Button(root, text="Compute YoY Changes", command=self.compute_yoy_changes, state=tk.DISABLED)
        self.compute_button.pack(pady=10)

        # Classify Rows Button
        self.classify_button = tk.Button(root, text="Classify Rows", command=self.classify_rows, state=tk.DISABLED)
        self.classify_button.pack(pady=10)

        # Visualize Clusters Button
        self.visualize_button = tk.Button(root, text="Visualize Clusters", command=self.visualize_clusters, state=tk.DISABLED)
        self.visualize_button.pack(pady=10)

        # Save Resulting CSV Button
        self.save_button = tk.Button(root, text="Save Resulting CSV", command=self.save_csv, state=tk.DISABLED)
        self.save_button.pack(pady=10)

        self.data = None

    def upload_csv(self):
        self.filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filename:
            try:
                self.data = pd.read_csv(self.filename)
                messagebox.showinfo("Success", "CSV file uploaded successfully!")
                self.compute_button.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV file: {e}")

    def compute_yoy_changes(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        try:
            selected_columns = [
                "Adjusted net national income per capita (current US$) [NY.ADJ.NNTY.PC.CD]",
                "GDP per capita (current US$) [NY.GDP.PCAP.CD]",
                "Consumer price index (2010 = 100) [FP.CPI.TOTL]",
                "Inflation, consumer prices (annual %) [FP.CPI.TOTL.ZG]"
            ]

            # Ensure 'Time' column exists
            if "Time" not in self.data.columns:
                messagebox.showerror("Error", "'Time' column is required in the dataset.")
                return

            self.data.sort_values(by=["Country Name", "Time"], inplace=True)
            
            for col in selected_columns:
                if col in self.data.columns:
                    yoy_col_name = f"YoY Change in {col}"
                    self.data[yoy_col_name] = self.data.groupby("Country Name")[col].pct_change() * 100

            messagebox.showinfo("Success", "Year-over-Year changes computed successfully!")
            self.save_button.config(state=tk.NORMAL)
            self.classify_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute YoY changes: {e}")

    def classify_rows(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        try:
            # Select all numeric columns except "Country Name" and "Time"
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col not in ["Time"]]

            # Drop rows with missing values in numeric columns
            data_for_clustering = self.data[numeric_columns].dropna()

            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_for_clustering)

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)

            # Map clusters to labels (Rising, Normal, Decreasing) based on centroid ordering
            cluster_centroids = kmeans.cluster_centers_.mean(axis=1)
            cluster_labels = {cluster: label for cluster, label in zip(np.argsort(cluster_centroids), ["Decreasing", "Normal", "Rising"])}
            
            # Assign classifications back to original data
            data_for_clustering["Cluster"] = clusters
            data_for_clustering["CoL Classification"] = data_for_clustering["Cluster"].map(cluster_labels)

            # Merge classifications into the original dataset
            self.data = self.data.merge(data_for_clustering["CoL Classification"], how="left", left_index=True, right_index=True)

            messagebox.showinfo("Success", "Rows classified successfully using KMeans clustering!")
            self.visualize_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to classify rows: {e}")

    def visualize_clusters(self):
        if self.data is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        try:
            # Select all numeric columns except "Country Name" and "Time"
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_columns = [col for col in numeric_columns if col not in ["Time"]]

            # Drop rows with missing values in numeric columns
            data_for_clustering = self.data[numeric_columns].dropna()

            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_for_clustering)

            # Apply PCA for 2D visualization
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_data)

            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)

            # Plot the PCA results with cluster labels
            plt.figure(figsize=(10, 6))
            plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.7)
            plt.title("Cluster Visualization (2D PCA)")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.colorbar(label="Cluster")
            plt.grid(True)
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize clusters: {e}")

    def save_csv(self):
        if self.data is None:
            messagebox.showerror("Error", "No data to save.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            try:
                self.data.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"CSV file saved successfully to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save CSV file: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CoLAnalysisApp(root)
    root.mainloop()
