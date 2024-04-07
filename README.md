# Handwriting-num-recognizing
Handwriting number recognizer based on LeNet5(PyTorch)

during the training, I used matplotlib to visualize the changing of the loss:
<img src=".\loss.png">
# Training
Run `train.py`<br> 
and if you want to custom training, you can just edit the last row of the file:`train(1,10)`<br>
The first parameter is pre_epochs, if it's set to zero, the code will start training a new network.If not, the code will use the pre-trained model.<br>
The second parameter is epochs, as the end epoch number.<br>

# Utilizing
Run `util.py`<br>
If you want to know the accuracy of your model, input "evaluate".<br>
If you want to use your own figure pictures, put your pictures under the folder 'input' and then input "predict" in the terminal.
