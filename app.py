import pandas as pd 
import pickle
import streamlit as st 

# Load model
model_knn = pickle.load(open('Model/book_recommend.pkl', 'rb'))

# Load books data
books = pd.read_csv('Dataset/BX-Books.csv',
                   encoding="ISO-8859-1",
                   sep=";",
                   header=0,
                   names=['isbn', 'title', 'author'],
                   usecols=['isbn', 'title', 'author'],
                   dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

# CRITICAL FIX: Match dataset size with model
model_size = model_knn._fit_X.shape[0]
books = books.head(model_size).reset_index(drop=True)

def recommend(book_title):
    book_title = book_title.lower()
    
    # Find books that contain the search term
    matching_books = books[books['title'].str.lower().str.contains(book_title, na=False)]
    
    if matching_books.empty:
        return []
    
    # Get the first match
    idx = matching_books.index[0]
    
    # Get recommendations
    distances, indices = model_knn.kneighbors([model_knn._fit_X[idx]], n_neighbors=6)
    
    recommended_books = []
    for i in range(1, len(indices[0])):  # Skip first (same book)
        recommended_books.append(books.iloc[indices[0][i]]['title'])
    
    return recommended_books

# Streamlit interface
st.title('📚 Book Recommendation System')
st.write("Type a book title you like and get similar books!")

# Show dataset info
st.info(f"Database contains {len(books)} books")

book_title = st.text_input('Book Title')

if st.button('Recommend'):
    if book_title:
        recommendations = recommend(book_title)
        if recommendations:
            st.subheader('You might also like:')
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.warning("Book not found. Please try another title.")
            # Show some sample books
            st.write("**Sample books in database:**")
            sample_books = books.head(10)['title'].tolist()
            for book in sample_books:
                st.write(f"• {book}")
    else:
        st.warning("Please enter a book title.")