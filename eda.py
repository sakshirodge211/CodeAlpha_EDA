import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("books_data.csv")

# Convert Price into numeric value
df["Price_Numeric"] = (
    df["Price"]
    .str.replace("£", "", regex=False)
    .astype(float)
)

# Convert rating words into numbers
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["Rating_Numeric"] = df["Rating"].map(rating_map)

# -----------------------------
# 1. Price Distribution
# -----------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["Price_Numeric"], bins=20, edgecolor="black")

plt.title("Book Price Distribution")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")

plt.tight_layout()
plt.savefig("price_distribution.png")
plt.show()


# -----------------------------
# 2. Rating Distribution
# -----------------------------

rating_counts = df["Rating_Numeric"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(
    rating_counts.index,
    rating_counts.values,
    edgecolor="black"
)

plt.title("Book Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Books")

plt.xticks([1, 2, 3, 4, 5])

plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()


# -----------------------------
# 3. Average Price by Rating
# -----------------------------

avg_price = df.groupby("Rating_Numeric")["Price_Numeric"].mean()

plt.figure(figsize=(8, 5))

plt.bar(
    avg_price.index,
    avg_price.values,
    edgecolor="black"
)

plt.title("Average Book Price by Rating")
plt.xlabel("Rating")
plt.ylabel("Average Price (£)")

plt.xticks([1, 2, 3, 4, 5])

plt.tight_layout()
plt.savefig("average_price_by_rating.png")
plt.show()


# -----------------------------
# 4. Top 10 Most Expensive Books
# -----------------------------

top_10 = df.nlargest(10, "Price_Numeric")

print("\nTop 10 Most Expensive Books:")
print(
    top_10[
        ["Title", "Price", "Rating"]
    ].to_string(index=False)
)

print("\nEDA Visualization Completed!")
print("Charts saved successfully.")


# -----------------------------
# Save analyzed dataset
# -----------------------------

df.to_csv("books_eda.csv", index=False, encoding="utf-8-sig")

print("Analyzed dataset saved as: books_eda.csv")