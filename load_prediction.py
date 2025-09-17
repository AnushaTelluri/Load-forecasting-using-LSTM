import tkinter as tk
from tkinter import filedialog, Label, Button, Text, Scrollbar, OptionMenu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import os

# ----- SETTINGS -----
SEQUENCE_LENGTH = 50  # same as training
MODEL_FILE = "LSTM_best_model.hdf5"  # your saved model
THRESHOLD = 3  # threshold for exceedances

def select_file_and_predict():
    def predict():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        # ---- Load CSV ----
        data = pd.read_csv(file_path, header=None)
        load_values = data.iloc[:, 1].astype(float).values

        percent = int(selected_percent.get())
        data_length = len(load_values)
        start_index = data_length - int(percent / 100 * data_length)
        selected_data = load_values[start_index:]

        # Ensure length fits sequence length
        remainder = len(selected_data) % SEQUENCE_LENGTH
        if remainder != 0:
            selected_data = selected_data[:-remainder]

        # ---- Convert to supervised matrix like training ----
        matrix_load = []
        for i in range(len(selected_data) - SEQUENCE_LENGTH + 1):
            matrix_load.append(selected_data[i:i + SEQUENCE_LENGTH])
        matrix_load = np.array(matrix_load)

        # ---- Scale like training ----
        scaler = StandardScaler()
        matrix_load_scaled = scaler.fit_transform(matrix_load)

        # ---- Split X, y exactly like training ----
        X = matrix_load_scaled[:, :-1]  # 49 time steps
        y = matrix_load_scaled[:, -1]

        # reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # ---- Load model ----
        if not os.path.exists(MODEL_FILE):
            output_text.config(state='normal')
            output_text.delete('1.0', tk.END)
            output_text.insert(tk.END, f"Error: Model file {MODEL_FILE} not found.\n")
            output_text.config(state='disabled')
            return

        model = load_model(MODEL_FILE)

        # ---- Predict ----
        pred_scaled = model.predict(X)

        # ---- Inverse transform to original scale ----
        def invert_scaling(X_scaled, y_scaled):
            combined = np.hstack([X_scaled, y_scaled])
            inverted = scaler.inverse_transform(combined)
            return inverted[:, -1]

        true_values = invert_scaling(X.reshape((X.shape[0], -1)), y.reshape(-1, 1))
        pred_values = invert_scaling(X.reshape((X.shape[0], -1)), pred_scaled)

        # ---- Time axis ----
        execution_time = datetime.now()
        hourly_datetime = [execution_time + timedelta(hours=i) for i in range(len(pred_values))]

        # ---- Find exceedances ----
        exceed_times = [(hourly_datetime[i], val) for i, val in enumerate(pred_values) if val > THRESHOLD]

        # ---- Clear old plot widgets ----
        for widget in window.pack_slaves():
            if isinstance(widget, tk.Widget):
                try:
                    if isinstance(widget, FigureCanvasTkAgg):
                        widget.get_tk_widget().destroy()
                except:
                    pass

        # ---- Plot ----
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(true_values, label='Actual Load')
        ax.plot(pred_values, label='Predicted Load')
        ax.axhline(THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD})')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Load')
        ax.set_title(f'Predicted Load vs Actual Load ({percent}% of Data)')
        ax.legend()
        ax.grid(True)

        # ---- Show text outputs ----
        output_text.config(state='normal')
        output_text.delete('1.0', tk.END)
        output_text.insert(tk.END, f'Hourly Date/Time\t\tPredicted Load\n')
        for i, val in enumerate(pred_values):
            output_text.insert(tk.END, f'{hourly_datetime[i].strftime("%Y-%m-%d %H:%M:%S")}\t\t{val:.4f}\n')
        output_text.config(state='disabled')

        exceed_text.config(state='normal')
        exceed_text.delete('1.0', tk.END)
        exceed_text.insert(tk.END, "Exceed Load Usage Time (Above Threshold):\n")
        exceed_text.insert(tk.END, "Date/Time\t\tLoad Value\n")
        for time, load in exceed_times:
            exceed_text.insert(tk.END, f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t\t{load:.4f}\n")
        exceed_text.config(state='disabled')

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # ---- GUI Layout ----
    window = tk.Tk()
    window.title("Load Prediction")

    Label(window, text="Select Percentage of Data for Prediction:").pack()
    options = [10, 20, 30, 80, 90, 100]
    selected_percent = tk.StringVar(window)
    selected_percent.set(options[0])
    percent_menu = OptionMenu(window, selected_percent, *options)
    percent_menu.pack()

    Button(window, text="Select File", command=predict).pack()

    output_text = Text(window, height=15, width=60)
    output_text.pack()

    exceed_text = Text(window, height=10, width=60)
    exceed_text.pack()

    scrollbar1 = Scrollbar(window, command=output_text.yview)
    scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)
    output_text.config(yscrollcommand=scrollbar1.set)

    scrollbar2 = Scrollbar(window, command=exceed_text.yview)
    scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
    exceed_text.config(yscrollcommand=scrollbar2.set)

    window.mainloop()

if __name__ == "__main__":
    select_file_and_predict()

