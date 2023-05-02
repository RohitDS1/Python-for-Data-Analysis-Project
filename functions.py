#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def factorial(n):
    return True


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

def eda(left_raw, right_raw):
    # Distribution of States in Left Dataset
    states = [state for state in left_raw['state'] if isinstance(state, str)]
    state_counts = pd.value_counts(states)
    labels, values = zip(*state_counts.items())
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title('Distribution of States in Left Dataset')
    plt.show()

    # Distribution of States in Right Dataset
    states = [state for state in right_raw['state'] if isinstance(state, str)]
    state_counts = pd.value_counts(states)
    labels, values = zip(*state_counts.items())
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title('Distribution of States in Right Dataset')
    plt.show()

    # Top 5 States with the Most Businesses in the Left Dataset
    state_counts = left_raw['state'].value_counts().nlargest(5)
    plt.figure(figsize=(8,6))
    plt.bar(state_counts.index, state_counts.values, color='purple')
    plt.title('Top 5 States with the Most Businesses (Left Dataset)', fontsize=16)
    plt.xlabel('State', fontsize=14)
    plt.ylabel('Number of Businesses', fontsize=14)
    plt.show()

    # Top 10 Cities with the Most Businesses in the Right Dataset
    city_counts = right_raw['city'].value_counts().nlargest(10)
    plt.figure(figsize=(18,6))
    plt.bar(city_counts.index, city_counts.values, color='orange')
    plt.title('Top 10 Cities with the Most Businesses (Right Dataset)', fontsize=16)
    plt.xlabel('City', fontsize=14)
    plt.ylabel('Number of Businesses', fontsize=14)
    plt.show()

    # Top 5 Categories in the Left Dataset
    all_categories = []
    for cats in left_raw['categories']:
        if isinstance(cats, str):
            all_categories.extend(cats.split(','))
    category_counts = {}
    for category in all_categories:
        category_counts[category.strip()] = category_counts.get(category.strip(), 0) + 1
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    labels = [x[0] for x in top_categories]
    values = [x[1] for x in top_categories]
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values)
    plt.title('Top 5 Categories (Left Dataset)', fontsize=16)
    plt.xlabel('Category', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.show()


# In[4]:


def matched_algo(A,B):
    
    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import py_stringsimjoin as ssj
    import py_stringmatching as sm
    import ssj
    import pandas as pd
    import os, sys
    from py_stringsimjoin.join.jaccard_join import jaccard_join
    from fuzzywuzzy import process 
    import re
    
    print('Number of records in A: ' + str(len(A)))
    print('Number of records in B: ' + str(len(B)))

    #A.entity_id, you are selecting the column with the label 'entity_id' from the DataFrame A.
    A.entity_id

    # Create a new column in DataFrame B called 'new_key_attr', and assign a range of integers
    # from 0 to the length of B to this column. This creates a unique identifier for each row in B
    # which can be used for matching with the corresponding rows in DataFrame A during a fuzzy join operation.
    B['new_key_attr'] = range(0, len(B))

    #Define a function to capitalize the first letter of each word in the name column
    def capitalize_name(name):
        name = name.str.lower().str.title()
        return name

    #Updating the returned result into the right dataset - name column
    B.loc[:, 'name'] = capitalize_name(B.name)

    # create whitespace tokenizer for tokenizing 'name' attribute. The return_set flag should be set to True since
    # Jaccard is a set based measure.
    ws = sm.WhitespaceTokenizer(return_set=True)

    # Use the ssj library to perform a fuzzy join between DataFrames A and B based on the 'name' attribute. The 'entity_id'
    # column from A will be matched against the 'business_id' column from B. The whitespace tokenizer object 'ws' will
    # be used to tokenize the 'name' attributes for both DataFrames. The Jaccard similarity threshold is set to 0.8,
    # meaning that pairs of rows with a Jaccard similarity score greater than or equal to 0.8 will be considered matches.
    # The output_pairs DataFrame will include the 'name' attribute for both A and B for all matched pairs.
    output_pairs_name = jaccard_join(A, B, 'entity_id', 'business_id', 'name', 'name', ws, 0.8,
                                    l_out_attrs=['name'], r_out_attrs=['name'])
    # Drop the '_id', 'l_name', and 'r_name' columns from the output_pairs DataFrame using the drop() method with the 'axis=1'
    # parameter to indicate that the columns should be dropped. The 'inplace=True' parameter ensures that the DataFrame is
    # modified in place rather than creating a new copy.
    output_pairs_name.drop(['_id', 'l_name', 'r_name'], axis=1, inplace=True)

    # Use the ssj library to perform a fuzzy join between DataFrames A and B based on the 'address' attribute. The 'entity_id'
    # column from A will be matched against the 'business_id' column from B. The whitespace tokenizer object 'ws' will
    # be used to tokenize the 'address' attributes for both DataFrames. The Jaccard similarity threshold is set to 0.8,
    # meaning that pairs of rows with a Jaccard similarity score greater than or equal to 0.8 will be considered matches.
    # The output_pairs_add DataFrame will include the 'address' attribute for both A and B for all matched pairs.
    output_pairs_address = jaccard_join(A, B, 'entity_id', 'business_id', 'address', 'address', ws, 0.8,
                                            l_out_attrs=['address'], r_out_attrs=['address'])
    # Remove the '_id', 'l_address', and 'r_address' columns from the output_pairs_add DataFrame since they are not needed.
    # The 'axis=1' parameter specifies that the columns should be dropped, and the 'inplace=True' parameter specifies
    # that the changes should be made to the DataFrame in place, without creating a new copy.
    output_pairs_address.drop(['_id', 'l_address', 'r_address'], axis=1, inplace=True)
    
    # concatenate the 2 data frames
    final_result_df = pd.concat([output_pairs_name, output_pairs_address])


    #Total number of matched records for both address and name
    print(f"Total number of matched records are: {final_result_df.count()}")
    
    # write the result data frame to a CSV file
    final_result_df.to_csv('FinalResult_MatchingRecords_Final.csv', index=False)

