<font face ="Arial">
First I used Dense layer with linear activation and second dense layer with softmax activation. <br>
Let 
v: vocablury_size   <br>
n: no_units_in_first_dense_layer.  <br>
W matrix: V x n is the center_word matrix. <br>
W' matrix: n x V is the context_word matrix.  <br>
 <br>
So output equation is <br>
<b>Y = softmax(W'* W* X).  </b> <br>
This is actually the probability of context_word given center_word.  <br>
and n is the size of word_embeddings. I have used n = 64.  <br>
 <br>
To plot the 2-D graph we have reduce diamentions of word_embeddings to 2.  <br><br>
<img src="./Figure_1.png" align="center" width = "50%"> 
</font>
