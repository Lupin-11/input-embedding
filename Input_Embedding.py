import torch
input_pair = torch.tensor([[  198, 21991,   761,   345,    11,   319,  6164, 15393],
        [  428,    11,  6622,   465,  3656,   416,   262,  3211],
        [ 5822,    26,   198,  1890,  6219,    11,   287,   262],
        [  198,  2504,   422,   262,  9403,  1108,   262,  6247]])
#input_pair is the data which is given to the llm as the input data.it has 8 block size.
#the given is a sample for the data which should be passed
batch_size=4#bach size can be modified.it is the no of batches we send at a time
block_size=8#block size can also be changed it is the no of blocks we send as input
num_tokens=50257#the number of tokens can be changed based on your project
vector_dimensions=256#this value can also be changed based on how many dimensions do we need
#creating an embedding layer dfor the vecrtr embedding
vector_embedding_meth=torch.nn.Embedding(num_tokens,vector_dimensions)
vector_embedding=vector_embedding_meth(input_pair)
print(vector_embedding)
#we are going to create a position embedding,in simple terms it means regards of the position of the word the meaning of the vector does not change.
# It also works on the embedding function of the pytorch module
#we are going to use the block_size and vector dimension to define the position embedding
position_embedding_func=torch.nn.Embedding(block_size,vector_dimensions)
position_embedding=position_embedding_func(torch.arange(block_size))
print(position_embedding.shape)
print(vector_embedding.shape)
#now we are going to make an input embedding which would be feed to our model,it is the sum of vector_embedding and positional_embedding
input_embedding=vector_embedding+position_embedding
print(input_embedding)
