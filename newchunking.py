from tqdm import tqdm
import pandas as pd
import re
from transformers import AutoTokenizer
from indicnlp.tokenize import sentence_tokenize

# Initialize the LaBSE tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")

# def split_nepali_sentences(text):
#     return sentence_tokenize.sentence_split(text, lang='ne')

def get_token_length(text):
    """Get the number of tokens in text using LaBSE tokenizer"""
    return len(tokenizer.encode(text)) - 2  # Subtract 2 for special tokens [CLS] and [SEP]

def chunk_text_by_tokens(text, max_tokens=512, overlap_ratio=0.2):
    """
    Chunk text while respecting token limits using LaBSE tokenizer
    
    Args:
        text (str): Input text to chunk
        max_tokens (int): Maximum tokens per chunk
        overlap_ratio (float): Ratio of overlap between chunks
    
    Returns:
        list: List of chunk texts
    """
    # Get all tokens at once
    all_tokens = tokenizer.encode(text)[1:-1]  # Remove [CLS] and [SEP]
    chunks = []
    
    # Calculate overlap size
    overlap_tokens = int(max_tokens * overlap_ratio)
    chunk_step = max_tokens - overlap_tokens
    
    # Create chunks with overlap
    for i in range(0, len(all_tokens), chunk_step):
        # Get chunk tokens
        chunk_tokens = all_tokens[i:i + max_tokens]
        
        # Decode chunk back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        if chunk_text.strip():
            chunks.append(chunk_text)
            
        # If remaining tokens are less than overlap size, stop
        if len(all_tokens) - (i + chunk_step) < overlap_tokens:
            break
    
    # Handle remaining tokens if any
    remaining_start = len(chunks) * chunk_step
    if remaining_start < len(all_tokens):
        remaining_tokens = all_tokens[remaining_start:]
        if len(remaining_tokens) > 0:
            remaining_text = tokenizer.decode(remaining_tokens, skip_special_tokens=True)
            if remaining_text.strip():
                chunks.append(remaining_text)
    
    return chunks

def remove_empty_faisala_details(df):
    """
    Removes rows where 'Faisala Detail' is NaN or empty.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with empty rows removed.
    """
    initial_rows = len(df)
    # Remove rows where 'Faisala Detail' is NaN or empty string
    df = df.dropna(subset=['Faisala Detail'])
    df = df[df['Faisala Detail'].str.strip() != '']
    removed_rows = initial_rows - len(df)
    print(f"Removed {removed_rows} rows with empty or NaN 'Faisala Detail' values")
    return df

def preprocess_and_chunk_faisala_Detail_with_sentences(file_path, output_file, max_tokens=512, overlap_ratio=0.2):
    """
    Preprocesses and chunks Faisala Detail using LaBSE tokenizer
    """
    # Load the CSV file
    df = pd.read_csv(file_path, dtype=str)

    # Ensure the Faisala Detail column exists
    if 'Faisala Detail' not in df.columns:
        raise ValueError("The input file must contain a 'Faisala Detail' column.")
    
    # Remove rows with empty Faisala Detail
    df = remove_empty_faisala_details(df)
    # df=df[:5]
    
    # Preprocess the Faisala Detail column
    def preprocess_text(text):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces, tabs, and line breaks with a single space
        return text.strip()

    df['Faisala Detail'] = df['Faisala Detail'].apply(preprocess_text)
    
    chunked_data = []
    for doc_idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Faisala Detail"):
        faisala_text = row['Faisala Detail']
        chunks = chunk_text_by_tokens(faisala_text, max_tokens, overlap_ratio)
        
        for chunk_idx, chunk_text in enumerate(chunks, 1):
            if chunk_text.strip():
                chunk_row = row.copy()
                chunk_row['Chunk Index'] = f"D{doc_idx}__C{chunk_idx}"
                chunk_row['Chunk Text'] = chunk_text
                chunk_row['Token Count'] = get_token_length(chunk_text)
                chunked_data.append(chunk_row)
    
    # Create DataFrame and save
    chunked_df = pd.DataFrame(chunked_data)
    chunked_df = chunked_df.drop(columns=['Faisala Detail'])
    
    # Reorder columns
    other_columns = [col for col in chunked_df.columns 
                    if col not in ['Subject', 'Chunk Index', 'Chunk Text', 'Token Count']]
    columns_order = other_columns + ['Subject', 'Chunk Index', 'Chunk Text', 'Token Count']
    chunked_df = chunked_df[columns_order]
    
    chunked_df.to_csv(output_file, index=False)
    print(f"✅ Chunking completed and saved to {output_file}!")

if __name__ == "__main__":
    input_file = "/home/abiral/Desktop/Project/NepSaul/abiral/data/nfc_normalized_aftersubjectseperation.csv"
    output_file = "/home/abiral/Desktop/Project/NepSaul/abiral/data/chunking_labse_tokens.csv"
    preprocess_and_chunk_faisala_Detail_with_sentences(
        input_file, 
        output_file, 
        max_tokens=500, 
        overlap_ratio=0.1
    )