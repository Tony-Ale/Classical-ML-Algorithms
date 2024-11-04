from collections import Counter
import numpy as np
def cosineSimilarity(text1:str, text2:str):
    """
    Sentence similarity using cosine similarity
    The closer cos_thetha is to 1 the higher the similarity
    """

    # Transform each word to a vector
    combined_text = text1 + " " + text2
    split_text = combined_text.split()
    freq_dict = dict(Counter(split_text))

    # create vectors for texts
    vector_text1 = np.zeros(shape=(len(freq_dict),))
    vector_text2 = np.zeros(shape=(len(freq_dict),))

    split_text1 = set(text1.split())
    split_text2 = set(text2.split())

    for idx, word in enumerate(freq_dict):
        if word in split_text1:
            vector_text1[idx] = freq_dict[word]
        if word in split_text2:
            vector_text2[idx] = freq_dict[word]

    # finding cosine similarity using dot product
    dot_product = np.dot(vector_text1, vector_text2)
    mag_vector1 = np.linalg.norm(vector_text1)
    mag_vector2 = np.linalg.norm(vector_text2)

    cos_thetha = dot_product/(mag_vector1 * mag_vector2)

    print(vector_text1, vector_text2)

    return cos_thetha

def similarity_euclidean_distance(text1, text2):
    """Sentence similarity using euclidean distance as metric"""
    # Transform each word to a vector
    combined_text = text1 + " " + text2
    split_text = combined_text.split()
    freq_dict = dict(Counter(split_text))

    # create vectors for texts
    vector_text1 = np.zeros(shape=(len(freq_dict),))
    vector_text2 = np.zeros(shape=(len(freq_dict),))

    split_text1 = set(text1.split())
    split_text2 = set(text2.split())

    for idx, word in enumerate(freq_dict):
        if word in split_text1:
            vector_text1[idx] = freq_dict[word]
        if word in split_text2:
            vector_text2[idx] = freq_dict[word]

    # calculating euclidean distance 
    distance = np.linalg.norm(vector_text1-vector_text2)

    # using the sum of the magnitude of the two vectors for normalization; not ideal, but it serves our use case
    mag_vector1 = np.linalg.norm(vector_text1)
    mag_vector2 = np.linalg.norm(vector_text2)

    sum_mag = mag_vector1 + mag_vector2

    prob = (sum_mag-distance)/sum_mag

    return prob




if __name__ == "__main__":
    result = cosineSimilarity("this is a completely different sentence", "this is a bit different though")

    result2 = similarity_euclidean_distance("this is a completely different sentence", "this is a bit different though")
    print(result)
    print(result2)





