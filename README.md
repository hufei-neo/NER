Simple Position Tagging model based on Glove, biLSTM and Mutil Head Attention<br/>
Loss is based on weighted SoftmaxCE<br/>

----------
Model Structures:<br/>
input <br /> 
↓<br /> 
embedding(Glove, FastText)<br /> 
↓<br /> 
biLSTM<br /> 
↓<br /> 
MHA(or not)<br />
↓<br /> 
FC<br /> 
↓<br /> 
Output<br />    