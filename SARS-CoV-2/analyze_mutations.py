import numpy as np

print('Welcome to the global vs. local sequence alignment competition!')

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """
    Performs global sequence alignment using the Needleman-Wunsch algorithm.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
        match (int): The score for a match.
        mismatch (int): The score for a mismatch.
        gap (int): The penalty for a gap.

    Returns:
        tuple: A tuple containing the alignment score, aligned sequence 1,
               and aligned sequence 2.
    """
    n = len(seq1)
    m = len(seq2)

    # Step 1: Initialize the scoring matrix
    # We use numpy for easier matrix operations
    score_matrix = np.zeros((m + 1, n + 1))

    # Fill the first row and column with gap penalties
    for i in range(m + 1):
        score_matrix[i][0] = i * gap
    for j in range(n + 1):
        score_matrix[0][j] = j * gap

    # Step 2: Fill the rest of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Calculate the score for a match or mismatch
            match_score = score_matrix[i - 1][j - 1] + (match if seq1[j - 1] == seq2[i - 1] else mismatch)
            
            # Calculate the score for a gap in seq1 (coming from above)
            gap_in_seq1 = score_matrix[i - 1][j] + gap
            
            # Calculate the score for a gap in seq2 (coming from the left)
            gap_in_seq2 = score_matrix[i][j - 1] + gap
            
            # Choose the maximum of the three scores
            score_matrix[i][j] = max(match_score, gap_in_seq1, gap_in_seq2)

    # Step 3: Traceback to find the optimal alignment
    align1, align2 = "", ""
    i, j = m, n
    
    while i > 0 and j > 0:
        current_score = score_matrix[i][j]
        diagonal_score = score_matrix[i - 1][j - 1]
        
        # Check if the path came from the diagonal
        if current_score == diagonal_score + (match if seq1[j - 1] == seq2[i - 1] else mismatch):
            align1 += seq1[j - 1]
            align2 += seq2[i - 1]
            i -= 1
            j -= 1
        # Check if the path came from above (gap in seq1)
        elif current_score == score_matrix[i - 1][j] + gap:
            align1 += '-'
            align2 += seq2[i - 1]
            i -= 1
        # The path must have come from the left (gap in seq2)
        else:
            align1 += seq1[j - 1]
            align2 += '-'
            j -= 1
            
    # Finish the traceback if we hit the top or left edge
    while j > 0:
        align1 += seq1[j - 1]
        align2 += '-'
        j -= 1
    while i > 0:
        align1 += '-'
        align2 += seq2[i - 1]
        i -= 1
        
    # The alignments are built backward, so we reverse them
    alignment_score = score_matrix[m][n]
    
    return alignment_score, align1[::-1], align2[::-1]

# --- Example Usage ---
seq_a = input('Please insert genome 1: ')
seq_b = input('Please insert genome 2: ')

score, aligned_a, aligned_b = needleman_wunsch(seq_a, seq_b)

# Create a string to show matches
match_str = ""
for char_a, char_b in zip(aligned_a, aligned_b):
    if char_a == char_b:
        match_str += "|"
    elif char_a == '-' or char_b == '-':
        match_str += " "
    else:
        match_str += "."

print(f"Global Alignment Score: {score}")
print(aligned_a)
print(match_str)
print(aligned_b)


#------------------------
#now compare it to local alignment

def smith_waterman(query_seq, target_seq, match_val=2, mismatch_val=-1, gap_val=-1):
    """
    Performs local sequence alignment using the Smith-Waterman algorithm.

    Args:
        query_seq (str): The first sequence (query).
        target_seq (str): The second sequence (target).
        match_val (int): The score for a match.
        mismatch_val (int): The penalty for a mismatch.
        gap_val (int): The penalty for a gap.

    Returns:
        tuple: A tuple containing the highest alignment score, the aligned
               query substring, and the aligned target substring.
    """
    # Use different variable names for dimensions to avoid conflicts
    rows = len(target_seq)
    cols = len(query_seq)
    
    # Step 1: Initialize the dynamic programming matrix
    # Also keep track of the position of the highest score
    dp_matrix = np.zeros((rows + 1, cols + 1))
    max_score = 0
    max_pos = (0, 0) # (row, col)

    # Step 2: Fill the matrix
    for row_idx in range(1, rows + 1):
        for col_idx in range(1, cols + 1):
            # Calculate scores for three possible moves
            diag_move = dp_matrix[row_idx - 1][col_idx - 1] + (match_val if query_seq[col_idx - 1] == target_seq[row_idx - 1] else mismatch_val)
            up_move = dp_matrix[row_idx - 1][col_idx] + gap_val
            left_move = dp_matrix[row_idx][col_idx - 1] + gap_val
            
            # The score cannot be negative, which allows a new alignment to start
            current_val = max(0, diag_move, up_move, left_move)
            dp_matrix[row_idx][col_idx] = current_val

            # Update the maximum score found so far
            if current_val > max_score:
                max_score = current_val
                max_pos = (row_idx, col_idx)

    # Step 3: Traceback from the cell with the highest score
    if max_score == 0:
        return 0, "", ""

    aligned_query, aligned_target = "", ""
    (current_row, current_col) = max_pos
    
    # Trace back until a score of 0 is encountered
    while dp_matrix[current_row][current_col] > 0:
        diag_score = dp_matrix[current_row - 1][current_col - 1]
        
        # Check if the path came from the diagonal
        if dp_matrix[current_row][current_col] == diag_score + (match_val if query_seq[current_col - 1] == target_seq[current_row - 1] else mismatch_val):
            aligned_query += query_seq[current_col - 1]
            aligned_target += target_seq[current_row - 1]
            current_row -= 1
            current_col -= 1
        # Check if the path came from above
        elif dp_matrix[current_row][current_col] == dp_matrix[current_row - 1][current_col] + gap_val:
            aligned_query += '-'
            aligned_target += target_seq[current_row - 1]
            current_row -= 1
        # The path must have come from the left
        else:
            aligned_query += query_seq[current_col - 1]
            aligned_target += '-'
            current_col -= 1

    # Return the final score and the reversed (corrected) alignment strings
    return max_score, aligned_query[::-1], aligned_target[::-1]

# --- Example Usage ---
# Two long sequences with a highly similar region embedded within them
seq_X = seq_a
seq_Y = seq_b

best_score, local_align_X, local_align_Y = smith_waterman(seq_X, seq_Y)

# Create a string to show matches
match_symbols = ""
for char_x, char_y in zip(local_align_X, local_align_Y):
    if char_x == char_y:
        match_symbols += "|"
    elif char_x == '-' or char_y == '-':
        match_symbols += " "
    else:
        match_symbols += "."

print(f"Local Alignment Score: {best_score}")
print(local_align_X)
print(match_symbols)
print(local_align_Y)