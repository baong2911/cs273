"""Minimal Tkinter GUI for classifying a new hotel review."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from src.predict import model_choices, predict_from_choice


class SentimentApp:
    """Small GUI shell; all ML work is delegated to ``src.predict``."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TripAdvisor Hotel Sentiment")
        self.root.geometry("700x500")
        self.choices = model_choices()
        self.choice_by_display = {choice["display_name"]: choice for choice in self.choices}

        main = ttk.Frame(root, padding=16)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="Enter a hotel review",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")
        self.review_box = tk.Text(main, height=12, wrap="word")
        self.review_box.pack(fill="both", expand=True, pady=(8, 14))

        ttk.Label(main, text="Model").pack(anchor="w")
        self.model_var = tk.StringVar(value=self.choices[0]["display_name"])
        self.model_dropdown = ttk.Combobox(
            main,
            textvariable=self.model_var,
            values=list(self.choice_by_display),
            state="readonly",
        )
        self.model_dropdown.pack(fill="x", pady=(4, 12))

        ttk.Button(main, text="Predict", command=self.predict).pack(anchor="w")
        self.result_var = tk.StringVar(value="")
        ttk.Label(
            main,
            textvariable=self.result_var,
            font=("TkDefaultFont", 14, "bold"),
        ).pack(anchor="w", pady=(16, 0))

    def predict(self) -> None:
        review = self.review_box.get("1.0", "end").strip()
        if not review:
            self.result_var.set("Please enter a review.")
            return
        choice = self.choice_by_display[self.model_var.get()]
        try:
            result = predict_from_choice(review, choice)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            messagebox.showerror("Prediction unavailable", str(exc))
            return
        sentiment = result["sentiment"].capitalize()
        details = ""
        if result.get("score_type") == "probability":
            details = f" (positive probability: {result['positive_probability']:.3f})"
        elif result.get("score_type") == "decision_function":
            details = f" (decision score: {result['decision_score']:.3f})"
        self.result_var.set(f"Prediction: {sentiment}{details}")


def main() -> None:
    root = tk.Tk()
    SentimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
